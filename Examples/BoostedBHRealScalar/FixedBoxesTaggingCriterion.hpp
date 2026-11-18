/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef FIXEDBOXESTAGGINGCRITERION_HPP_
#define FIXEDBOXESTAGGINGCRITERION_HPP_

#include "Cell.hpp"
#include "Coordinates.hpp"
#include "DimensionDefinitions.hpp"
#include "Tensor.hpp"

//! Fixed refinement boxes centred on the BH, with a user-chosen half-width
//! for every level. On level l it tags the cube max(|x|,|y|,|z|) < a_half_width
//! (measured from a_center), which then becomes level l + 1.
//! FixedGridsTaggingCriterion is the special case a_half_width = L 2^-(l+2),
//! which puts N cells across every level; here the fine levels can be kept
//! small around the BH while level 1 still covers the shock region.
class FixedBoxesTaggingCriterion
{
  protected:
    const double m_dx;
    const double m_half_width;
    const std::array<double, CH_SPACEDIM> m_center;

  public:
    FixedBoxesTaggingCriterion(const double a_dx, const double a_half_width,
                               const std::array<double, CH_SPACEDIM> a_center)
        : m_dx(a_dx), m_half_width(a_half_width), m_center(a_center){};

    template <class data_t> void compute(Cell<data_t> current_cell) const
    {
        data_t criterion = 0.0;
        const Coordinates<data_t> coords(current_cell, m_dx, m_center);
        const data_t max_abs_xy = simd_max(abs(coords.x), abs(coords.y));
        const data_t max_abs_xyz = simd_max(max_abs_xy, abs(coords.z));
        auto regrid = simd_compare_lt(max_abs_xyz, m_half_width);
        criterion = simd_conditional(regrid, 100.0, criterion);

        // Write back into the flattened Chombo box
        current_cell.store_vars(criterion, 0);
    }
};

#endif /* FIXEDBOXESTAGGINGCRITERION_HPP_ */
