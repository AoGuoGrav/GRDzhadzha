/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef SCALARPOTENTIAL_HPP_
#define SCALARPOTENTIAL_HPP_

#include "InitialScalarData.hpp"
#include "simd.hpp"

class ScalarPotential
{
  protected:
    const InitialScalarData::params_t m_initial_params;

  public:
    //! The constructor
    ScalarPotential(const InitialScalarData::params_t a_initial_params)
        : m_initial_params(a_initial_params)
    {
    }

    //! Set the potential function for the scalar field here
    template <class data_t, template <typename> class vars_t>
    void compute_potential(data_t &V_of_phi, data_t &dVdphi,
                           const vars_t<data_t> &vars) const
    {
        // The potential value at phi
        // 1/2 m^2 phi^2 + 1/4 lambda phi^4
        const double mu = m_initial_params.mass;
        const double lambda = m_initial_params.lambda;
        const data_t phi2 = vars.phi * vars.phi;
        V_of_phi = 0.5 * mu * mu * phi2 + 0.25 * lambda * phi2 * phi2;

        // The potential gradient at phi wrt the field
        // m^2 phi + lambda phi^3
        dVdphi = (mu * mu + lambda * phi2) * vars.phi;
    }
};

#endif /* SCALARPOTENTIAL_HPP_ */
