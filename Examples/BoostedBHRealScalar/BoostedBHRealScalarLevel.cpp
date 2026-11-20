/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

// General includes common to most GR problems
#include "BoostedBHRealScalarLevel.hpp"
#include "AMRReductions.hpp"
#include "BoxLoops.hpp"
#include "ComputePack.hpp"
#include "NanCheck.hpp"
#include "SetValue.hpp"
#include "SmallDataIO.hpp"
#include <algorithm>
#include <functional>

// For RHS update
#include "BoostedBH.hpp"
#include "MatterEvolution.hpp"

// For tag cells
#include "FixedBoxesTaggingCriterion.hpp"

// Problem specific includes
#include "EnergyConservation.hpp"
#include "ExcisionDiagnostics.hpp"
#include "ExcisionEvolution.hpp"
#include "FluxExtraction.hpp"
#include "InitialScalarData.hpp"
#include "LinearMomConservation.hpp"
#include "ScalarField.hpp"
#include "ScalarPotential.hpp"

// Initial data for field and metric variables
void BoostedBHRealScalarLevel::initialData()
{
    CH_TIME("BoostedBHRealScalarLevel::initialData");
    if (m_verbosity)
        pout() << "BoostedBHRealScalarLevel::initialData " << m_level << endl;

    // First set everything to zero, then set the value of the conformal factor
    // This is just for the diagnostics
    SetValue set_zero(0.0);
    BoostedBH boosted_bh(m_p.bg_params_at(m_time), m_dx); // just calculates chi
    auto compute_pack = make_compute_pack(set_zero, boosted_bh);
    BoxLoops::loop(compute_pack, m_state_diagnostics, m_state_diagnostics,
                   SKIP_GHOST_CELLS);

    // Now set the actual evolution variables
    InitialScalarData initial_sf(m_p.initial_params);
    BoxLoops::loop(initial_sf, m_state_new, m_state_new, FILL_GHOST_CELLS);

    // excise evolution vars within horizon, turn off simd vectorisation
    BoxLoops::loop(ExcisionEvolution<ScalarFieldWithPotential, BoostedBH>(
                       m_dx, m_p.center, boosted_bh),
                   m_state_new, m_state_new, SKIP_GHOST_CELLS, disable_simd());
}

void BoostedBHRealScalarLevel::specificPostTimeStep()
{
    // Check for nans on every level
    if (m_p.nan_check)
        BoxLoops::loop(NanCheck(), m_state_new, m_state_new, SKIP_GHOST_CELLS,
                       disable_simd());

    // At any level, but after the timestep on the minimum extraction level
    int min_level = 0;
    if (m_p.activate_extraction == 1)
        min_level = m_p.extraction_params.min_extraction_level();

    bool calculate_diagnostics = at_level_timestep_multiple(min_level);
    if (calculate_diagnostics)
    {
        fillAllGhosts();
        ScalarPotential potential(m_p.initial_params);
        ScalarFieldWithPotential scalar_field(potential);
        BoostedBH boosted_bh(m_p.bg_params_at(m_time), m_dx);
        EnergyConservation<ScalarFieldWithPotential, BoostedBH> energies(
            scalar_field, boosted_bh, m_dx, m_p.center);
        int direction = 0; // we want the x direction for the momentum
        LinearMomConservation<ScalarFieldWithPotential, BoostedBH>
            linear_momenta(scalar_field, boosted_bh, direction, m_dx,
                           m_p.center);
        BoxLoops::loop(make_compute_pack(energies, linear_momenta), m_state_new,
                       m_state_diagnostics, SKIP_GHOST_CELLS);

        // excise within/outside specified radii, no simd
        if (m_p.activate_excision == 1)
        {
            BoxLoops::loop(
                ExcisionDiagnostics<ScalarFieldWithPotential, BoostedBH>(
                    m_dx, m_p.center, boosted_bh, m_p.inner_r, m_p.outer_r),
                m_state_diagnostics, m_state_diagnostics, SKIP_GHOST_CELLS,
                disable_simd());
        }
    }

    // write out the integral after each timestep on the min_level
    if (m_p.activate_excision == 1)
    {
        if (m_level == min_level)
        {
            bool first_step = (m_time == m_dt);
            // integrate the densities and write to a file
            AMRReductions<VariableType::diagnostic> amr_reductions(m_gr_amr);
            double rhoEnergy_sum = amr_reductions.sum(c_rhoEnergy);
            double rhoLinMom_sum = amr_reductions.sum(c_rhoLinMom);
            double sourceLinMom_sum = amr_reductions.sum(c_sourceLinMom);

            SmallDataIO integral_file(m_p.data_path + "EnergyIntegrals", m_dt,
                                      m_time, m_restart_time,
                                      SmallDataIO::APPEND, first_step);
            // remove any duplicate data if this is post restart
            integral_file.remove_duplicate_time_data();

            std::vector<double> data_for_writing = {
                rhoEnergy_sum, rhoLinMom_sum, sourceLinMom_sum,
                m_p.bh_velocity_at(m_time)};

            // write data
            if (first_step)
            {
                integral_file.write_header_line(
                    {"Energy density.", "Lin. Mom. density", "Lin. Mom. source",
                     "BH velocity"});
            }
            integral_file.write_time_data_line(data_for_writing);

            if (m_p.num_shell_radii > 0)
                writeShellIntegrals(min_level, first_step);
        }
    }

    if (m_p.activate_extraction == 1)
    {
        if (m_level == min_level)
        {
            // Now refresh the interpolator and do the interpolation
            // only fill the actual ghost cells needed to save time
            bool fill_ghosts = false;
            m_gr_amr.m_interpolator->refresh(fill_ghosts);
            m_gr_amr.fill_multilevel_ghosts(
                VariableType::diagnostic, Interval(c_fluxEnergy, c_fluxLinMom));
            FluxExtraction my_extraction(m_p.extraction_params, m_dt, m_time,
                                         m_restart_time);
            my_extraction.execute_query(m_gr_amr.m_interpolator, m_p.data_path);
        }
    }
}

// Volume integrals of rhoEnergy, rhoLinMom and sourceLinMom over the nested
// shells inner_r < r < R for every R in shell_radii. The shells are done from
// the largest to the smallest: each pass zeroes the diagnostics outside R on
// all levels >= a_min_level and sums them again. This destroys the
// diagnostics outside the smallest shell, which is harmless because every
// level >= a_min_level recomputes them before the next reduction (and the
// diagnostics are not plotted); the coarser levels are not touched, and the
// shells lie inside the refinement box of a_min_level, where they are covered.
void BoostedBHRealScalarLevel::writeShellIntegrals(const int a_min_level,
                                                   const bool a_first_step)
{
    std::vector<double> radii = m_p.shell_radii;
    std::sort(radii.begin(), radii.end(), std::greater<double>());
    const int num_shells = radii.size();

    AMRReductions<VariableType::diagnostic> amr_reductions(m_gr_amr);
    std::vector<double> QE(num_shells), Qx(num_shells), S(num_shells);
    for (int ishell = 0; ishell < num_shells; ++ishell)
    {
        for (GRAMRLevel *level : m_gr_amr.get_gramrlevels())
        {
            if (level->level() < a_min_level)
                continue;
            const double dx = level->get_dx();
            BoostedBH boosted_bh(m_p.bg_params_at(m_time), dx);
            GRLevelData &diagnostics = const_cast<GRLevelData &>(
                level->getLevelData(VariableType::diagnostic));
            BoxLoops::loop(
                ExcisionDiagnostics<ScalarFieldWithPotential, BoostedBH>(
                    dx, m_p.center, boosted_bh, m_p.inner_r, radii[ishell]),
                diagnostics, diagnostics, SKIP_GHOST_CELLS, disable_simd());
        }
        QE[ishell] = amr_reductions.sum(c_rhoEnergy);
        Qx[ishell] = amr_reductions.sum(c_rhoLinMom);
        S[ishell] = amr_reductions.sum(c_sourceLinMom);
    }

    // write from the smallest to the largest shell
    std::vector<double> data_for_writing;
    std::vector<std::string> header;
    for (int ishell = num_shells - 1; ishell >= 0; --ishell)
    {
        data_for_writing.push_back(QE[ishell]);
        data_for_writing.push_back(Qx[ishell]);
        data_for_writing.push_back(S[ishell]);
        const std::string r = std::to_string(static_cast<int>(radii[ishell]));
        header.push_back("Q_E(r<" + r + ")");
        header.push_back("Q_x(r<" + r + ")");
        header.push_back("int S_x(r<" + r + ")");
    }

    SmallDataIO shell_file(m_p.data_path + "ShellIntegrals", m_dt, m_time,
                           m_restart_time, SmallDataIO::APPEND, a_first_step);
    shell_file.remove_duplicate_time_data();
    if (a_first_step)
        shell_file.write_header_line(header);
    shell_file.write_time_data_line(data_for_writing);
}

// Things to do in RHS update, at each RK4 step
void BoostedBHRealScalarLevel::specificEvalRHS(GRLevelData &a_soln,
                                               GRLevelData &a_rhs,
                                               const double a_time)
{
    // Calculate right hand side with matter_t = ScalarField
    // and background_t = BoostedBH
    ScalarPotential potential(m_p.initial_params);
    ScalarFieldWithPotential scalar_field(potential);
    BoostedBH boosted_bh(m_p.bg_params_at(a_time), m_dx);
    MatterEvolution<ScalarFieldWithPotential, BoostedBH> my_evolution(
        scalar_field, boosted_bh, m_p.sigma, m_dx, m_p.center);
    BoxLoops::loop(my_evolution, a_soln, a_rhs, SKIP_GHOST_CELLS);

    // Do excision within horizon, don't use vectorisation
    BoxLoops::loop(ExcisionEvolution<ScalarFieldWithPotential, BoostedBH>(
                       m_dx, m_p.center, boosted_bh),
                   a_soln, a_rhs, SKIP_GHOST_CELLS, disable_simd());
}

// Note that for the fixed grids this only happens on the initial timestep
void BoostedBHRealScalarLevel::computeTaggingCriterion(
    FArrayBox &tagging_criterion, const FArrayBox &current_state)
{
    // cells tagged on this level form level m_level + 1, a cube of half-width
    // level_half_widths[m_level] around the BH (no tags on the finest level)
    const double half_width =
        (m_level < m_p.max_level) ? m_p.level_half_widths[m_level] : 0.0;
    BoxLoops::loop(
        FixedBoxesTaggingCriterion(m_dx, half_width, m_p.bg_params.center),
        current_state, tagging_criterion);
}
