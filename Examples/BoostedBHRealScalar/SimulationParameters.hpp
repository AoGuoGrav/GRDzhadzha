/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef SIMULATIONPARAMETERS_HPP_
#define SIMULATIONPARAMETERS_HPP_

// General includes
#include "FixedBGSimulationParametersBase.hpp"
#include "GRParmParse.hpp"

// Problem specific includes:
#include "BoostedBH.hpp"
#include "InitialScalarData.hpp"
#include <algorithm>

class SimulationParameters : public FixedBGSimulationParametersBase
{
  public:
    SimulationParameters(GRParmParse &pp) : FixedBGSimulationParametersBase(pp)
    {
        // read the problem specific params
        read_params(pp);
        check_params();
    }

    void read_params(GRParmParse &pp)
    {
        // Initial SF
        pp.load("scalar_amplitude", initial_params.amplitude, 0.1);
        pp.load("scalar_mass", initial_params.mass, 0.2);
        pp.load("scalar_lambda", initial_params.lambda, 0.0);

        // BH data
        pp.load("bh_mass", bg_params.mass, 1.0);
        pp.load("bh_velocity", bg_params.velocity, 0.0);
        pp.load("bh_center", bg_params.center, center);

        // optional smooth ramp of the BH velocity from bh_velocity to
        // bh_velocity_final, between ramp_start and ramp_start + ramp_duration
        pp.load("bh_velocity_final", bh_velocity_final, bg_params.velocity);
        pp.load("ramp_start", ramp_start, 0.0);
        pp.load("ramp_duration", ramp_duration, 0.0);

        // half-width of the fixed refinement cube of levels 1..max_level
        // around the BH; the default L 2^-(l+1) for level l reproduces
        // FixedGridsTaggingCriterion (N cells across every level)
        std::vector<double> default_half_widths(std::max(max_level, 1));
        for (int ilev = 0; ilev < max_level; ++ilev)
            default_half_widths[ilev] = L * pow(2.0, -(ilev + 2.0));
        pp.load("level_half_widths", level_half_widths,
                std::max(max_level, 1), default_half_widths);

        // optional extra volume integrals over the nested shells
        // inner_r < r < shell_radii[k] (each <= outer_r), written to
        // data/ShellIntegrals.dat, to see how the drag grows with the
        // size of the integration region
        pp.load("num_shell_radii", num_shell_radii, 0);
        if (num_shell_radii > 0)
            pp.load("shell_radii", shell_radii, num_shell_radii);
    }

    //! The BH velocity at time a_time, a C^2 step between the initial and
    //! final velocities (constant if ramp_duration <= 0)
    double bh_velocity_at(const double a_time) const
    {
        if (ramp_duration <= 0.0)
            return bg_params.velocity;
        const double s =
            std::min(std::max((a_time - ramp_start) / ramp_duration, 0.0), 1.0);
        const double step = s * s * s * (10.0 - 15.0 * s + 6.0 * s * s);
        return bg_params.velocity +
               (bh_velocity_final - bg_params.velocity) * step;
    }

    //! The background params at time a_time. The metric is the stationary
    //! boosted BH with the instantaneous velocity, so dgamma_ij/dt ~ dv/dt
    //! is neglected in K_ij, which is fine for a slow ramp
    BoostedBH::params_t bg_params_at(const double a_time) const
    {
        BoostedBH::params_t params = bg_params;
        params.velocity = bh_velocity_at(a_time);
        return params;
    }

    void check_params()
    {
        // frequency of the homogeneous field, to first order in
        // lambda phi0^2 / mu^2: omega^2 = mu^2 + 3/4 lambda phi0^2
        const double omega =
            sqrt(initial_params.mass * initial_params.mass +
                 0.75 * initial_params.lambda * initial_params.amplitude *
                     initial_params.amplitude);
        warn_parameter("scalar_mass", initial_params.mass,
                       omega < 0.2 / coarsest_dx / dt_multiplier,
                       "oscillations of scalar field do not appear to be "
                       "resolved on coarsest level");
        warn_parameter("scalar_lambda", initial_params.lambda,
                       initial_params.lambda >= 0.0,
                       "should be >= 0.0 (repulsive self-interaction)");
        warn_parameter("bh_mass", bg_params.mass, bg_params.mass >= 0.0,
                       "should be >= 0.0");
        check_parameter("bh_velocity_final", bh_velocity_final,
                        std::abs(bh_velocity_final) < 1.0,
                        "must be in the range -1.0 < velocity < 1.0");
        warn_parameter("ramp_duration", ramp_duration,
                       ramp_duration > 0.0 ||
                           bh_velocity_final == bg_params.velocity,
                       "should be > 0 for the velocity to reach "
                       "bh_velocity_final");
        FOR(idir)
        {
            std::string name = "bh_center[" + std::to_string(idir) + "]";
            warn_parameter(
                name, bg_params.center[idir],
                (bg_params.center[idir] >= 0) &&
                    (bg_params.center[idir] <= (ivN[idir] + 1) * coarsest_dx),
                "should be within the computational domain");
        }
        for (int ilev = 0; ilev < max_level; ++ilev)
        {
            std::string name =
                "level_half_widths[" + std::to_string(ilev) + "]";
            // each level must sit inside the one below it, with a margin
            // of a few coarser cells for proper nesting
            const double outer = (ilev == 0) ? 0.5 * L
                                             : level_half_widths[ilev - 1];
            const double coarser_dx = coarsest_dx * pow(2.0, -ilev);
            warn_parameter(name, level_half_widths[ilev],
                           (level_half_widths[ilev] > 0.0) &&
                               (level_half_widths[ilev] <=
                                outer - 4.0 * coarser_dx),
                           "should be positive and nested inside the level "
                           "below with a margin of a few cells");
        }
        if (num_shell_radii > 0)
        {
            check_parameter("activate_excision", activate_excision,
                            activate_excision,
                            "must be on for the shell_radii integrals");
            // the shell integrals only re-excise levels >= min_level, so the
            // shells must lie inside the refinement box of min_level
            const int min_level =
                activate_extraction
                    ? extraction_params.min_extraction_level()
                    : 0;
            const double box_half_width =
                (min_level > 0) ? level_half_widths[min_level - 1] : 0.5 * L;
            for (int i = 0; i < num_shell_radii; ++i)
            {
                std::string name = "shell_radii[" + std::to_string(i) + "]";
                check_parameter(name, shell_radii[i],
                                (shell_radii[i] > inner_r) &&
                                    (shell_radii[i] <= outer_r),
                                "must be > inner_r and <= outer_r");
                warn_parameter(name, shell_radii[i],
                               shell_radii[i] <= box_half_width -
                                                     4.0 * coarsest_dx *
                                                         pow(2.0, -min_level),
                               "should lie inside the refinement box of the "
                               "minimum extraction level");
            }
        }
    }

    // Collection of parameters necessary for the initial conditions
    InitialScalarData::params_t initial_params;
    // Collection of parameters necessary for the metric background
    // (bg_params.velocity is the initial velocity of the ramp)
    BoostedBH::params_t bg_params;
    double bh_velocity_final, ramp_start, ramp_duration;
    //! half-width of the refinement cube of level l + 1 around the BH
    std::vector<double> level_half_widths;
    //! outer radii of the extra integration shells inner_r < r < R
    int num_shell_radii;
    std::vector<double> shell_radii;
};

#endif /* SIMULATIONPARAMETERS_HPP_ */
