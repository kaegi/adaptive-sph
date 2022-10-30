use crate::{
    floating_type_mod::{FRAC_1_PI, FT, PI},
    simulation_parameters::SimulationParams,
    V2, V3, VF, VI,
};

#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(unused)]
pub enum ParticleSizes {
    Uniform,
    Adaptive,
}

#[cfg(feature = "uniform-particle-sizes")]
pub const PARTICLE_SIZES: ParticleSizes = ParticleSizes::Uniform;

#[cfg(not(feature = "uniform-particle-sizes"))]
pub const PARTICLE_SIZES: ParticleSizes = ParticleSizes::Adaptive;

/**
 *
 */
pub fn cubic_kernel_unnormalized(q: FT) -> FT {
    if q < 0.5 {
        return 6. * (q * q * q - q * q) + 1.;
    } else if q < 1. {
        let v = 1. - q;
        return 2. * (v * v * v);
    } else {
        return 0.;
    }
}

pub fn cubic_kernel_unnormalized_deriv(q: FT) -> FT {
    if q < 0.5 {
        return 18. * q * q - 12. * q;
    } else if q < 1. {
        let v = 1. - q;
        return -6. * v * v;
    } else {
        return 0.;
    };
}

/**
 * r is the distance to the center.
 * h is the support radius and smoothing length.
 */
pub fn cubic_kernel_2d(r: FT, h: FT) -> FT {
    let norm_factor = 10. / (7. * PI * (h * h));
    return norm_factor * cubic_kernel_unnormalized(r / (2. * h));
}
pub fn cubic_kernel_3d(r: FT, h: FT) -> FT {
    let norm_factor = 1. / (PI * (h * h * h));
    return norm_factor * cubic_kernel_unnormalized(r / (2. * h));
}

/**
 * Calculate the derivative dW/dx where W=kernel(|x-y|/h) and x-y=diff.
 */
pub fn cubic_kernel_2d_deriv(mut diff: V2, h: FT) -> V2 {
    let r = diff.norm();
    let q: FT = r / (2. * h);
    if q <= 1.0e-5 {
        return V2::zeros();
    }
    diff.unscale_mut(r);

    let norm_factor = 10. / (7. * PI * (h * h));
    return norm_factor * cubic_kernel_unnormalized_deriv(q) / (2. * h) * diff;
}

/**
 * Calculate the derivative dW/dx where W=kernel(|x-y|/h) and x-y=diff.
 */
fn cubic_kernel_3d_deriv(mut diff: V3, h: FT) -> V3 {
    let r: FT = diff.normalize_mut();
    let q: FT = r / (2. * h);
    if q <= 1.0e-5 {
        return V3::zeros();
    }
    diff.unscale_mut(r);

    let norm_factor = 1. / (PI * (h * h * h));
    return norm_factor * cubic_kernel_unnormalized_deriv(q) / (2. * h) * diff;
}

#[test]
fn cubic_kernel_2d_integration_test() {
    use crate::vec2f;

    let h = 5.;
    let support_radius = 2.0 * h;
    let grid_size = 200;
    let square_len = 2. * support_radius / grid_size as FT;
    let square_area = square_len * square_len;

    let mut integral = 0.;

    for y in 0..grid_size {
        for x in 0..grid_size {
            let integration_point = vec2f(
                (x as FT + 0.5) * square_len - support_radius,
                (y as FT + 0.5) * square_len - support_radius,
            );
            integral += cubic_kernel_2d(integration_point.norm(), h) * square_area;
        }
    }

    let allow_deviation = 1.00001;
    println!("Integration of 2D cubic kernel with h={:.2}: {}", h, integral);
    assert!(1.0 / allow_deviation <= integral);
    assert!(integral <= allow_deviation / 1.0);
}

#[test]
fn cubic_kernel_2d_derivative_test() {
    use crate::vec2f;

    let h = 5.;
    let support_radius = 2. * h;
    let test_grid_size = 100;
    let diff = support_radius * 1e-2;
    let diff_half = diff * 0.5;

    let probe_offset = 2. * support_radius / test_grid_size as FT;

    fn v2_str(v: &V2) -> String {
        format!("[{:+.7}, {:+.7}]", v.x, v.y)
    }

    for y in 0..=test_grid_size {
        for x in 0..=test_grid_size {
            let probe_point = vec2f(
                (x as FT + 0.5) * probe_offset - support_radius,
                (y as FT + 0.5) * probe_offset - support_radius,
            );

            let analytical_deriv = cubic_kernel_2d_deriv(probe_point, h);

            let x_neg: FT = cubic_kernel_2d((probe_point + vec2f(-diff_half, 0.)).norm(), h);
            let x_pos: FT = cubic_kernel_2d((probe_point + vec2f(diff_half, 0.)).norm(), h);
            let y_neg: FT = cubic_kernel_2d((probe_point + vec2f(0., -diff_half)).norm(), h);
            let y_pos: FT = cubic_kernel_2d((probe_point + vec2f(0., diff_half)).norm(), h);

            let approx_deriv = vec2f((x_pos - x_neg) / diff, (y_pos - y_neg) / diff);
            let absolute_error = analytical_deriv - approx_deriv;

            println!(
                "[{}, {}] at {}: analytical={} approx={} difference={}",
                x,
                y,
                v2_str(&probe_point),
                v2_str(&analytical_deriv),
                v2_str(&approx_deriv),
                v2_str(&(analytical_deriv - approx_deriv))
            );

            assert!(absolute_error.x.abs() < 0.001);
            assert!(absolute_error.y.abs() < 0.001);
        }
    }
}

// Sync is needed since we use this trait inside parallel iterators
pub trait DimensionUtils<const D: usize>: Sync {
    fn iterate_grid_neighbors(dist: i32, f: impl FnMut(VI<D>));

    fn kernelh(diff: VF<D>, h: FT) -> FT;
    fn kernel_derivh(diff: VF<D>, h: FT) -> VF<D>;

    /// This value is used for the technique "Constrained Neighbor Lists for SPH-based Fluid Simulations".
    fn support_radius_by_smoothing_length() -> FT;

    fn sphere_volume_to_radius(volume: FT) -> FT;
    fn radius_to_sphere_volume(r: FT) -> FT;
}

#[allow(dead_code)]
pub enum DimensionUtils2d {}
impl DimensionUtils<2> for DimensionUtils2d {
    fn iterate_grid_neighbors(dist: i32, mut f: impl FnMut(VI<2>)) {
        for y in -dist..=dist {
            for x in -dist..=dist {
                f([x, y].into());
            }
        }
    }

    fn kernelh(diff: VF<2>, h: FT) -> FT {
        cubic_kernel_2d(diff.norm(), h)
    }

    fn kernel_derivh(diff: VF<2>, h: FT) -> VF<2> {
        cubic_kernel_2d_deriv(diff, h)
    }

    fn support_radius_by_smoothing_length() -> FT {
        2.
    }

    /** In this 2D case it is the "circle area to radius" */
    fn sphere_volume_to_radius(area: FT) -> FT {
        // A = PI * r^2   =>  r = sqrt(A/PI)
        (area * FRAC_1_PI).sqrt()
    }

    /** In this 2D case it is the "radius to circle area " */
    fn radius_to_sphere_volume(r: FT) -> FT {
        PI * r * r
    }
}

#[test]
fn test_radius_and_sphere_volume_conversion() {
    fn inner<DU: DimensionUtils<D>, const D: usize>() {
        for x in [0.1, 0.5, 1.0, 100.] {
            let x2 = DU::radius_to_sphere_volume(DU::sphere_volume_to_radius(x));
            crate::assert_ft_approx_eq(x, x2, 0.000001, || {
                format!("roundtrip sphere_volume->radius->sphere_volume")
            });
        }
    }

    inner::<DimensionUtils2d, 2>();
    inner::<DimensionUtils2d, 3>();
}

#[allow(dead_code)]
pub enum DimensionUtils3d {}
impl DimensionUtils<3> for DimensionUtils2d {
    fn iterate_grid_neighbors(dist: i32, mut f: impl FnMut(VI<3>)) {
        for z in -dist..=dist {
            for y in -dist..=dist {
                for x in -dist..=dist {
                    f([x, y, z].into());
                }
            }
        }
    }

    fn kernelh(diff: VF<3>, h: FT) -> FT {
        cubic_kernel_3d(diff.norm(), h)
    }

    fn kernel_derivh(diff: VF<3>, h: FT) -> VF<3> {
        cubic_kernel_3d_deriv(diff, h)
    }

    fn support_radius_by_smoothing_length() -> FT {
        2.
    }

    fn sphere_volume_to_radius(volume: FT) -> FT {
        // V = 4PI/3 * r^3   =>   r = (3V/(4PI))^(1/3)
        (volume * (FRAC_1_PI * 3. * 0.25)).powf(1. / 3.)
    }

    fn radius_to_sphere_volume(r: FT) -> FT {
        4. * PI / 3. * r * r * r
    }
}

/** Determine kernel support radius */
pub fn smoothing_length_single(h: &[FT], i: usize, params: SimulationParams) -> FT {
    match PARTICLE_SIZES {
        ParticleSizes::Adaptive => h[i],
        ParticleSizes::Uniform => params.h,
    }
}

/** Determine kernel support radius */
pub fn smoothing_length(h: &[FT], i: usize, j: usize, params: SimulationParams) -> FT {
    match PARTICLE_SIZES {
        ParticleSizes::Adaptive => (h[i] + h[j]) * 0.5,
        ParticleSizes::Uniform => params.h,
    }
}

/** Determine kernel support radius */
pub fn support_radius<DU: DimensionUtils<D>, const D: usize>(
    h: &[FT],
    i: usize,
    j: usize,
    params: SimulationParams,
) -> FT {
    smoothing_length(h, i, j, params) * DU::support_radius_by_smoothing_length()
}

/** Determine kernel support radius */
pub fn support_radius_single<DU: DimensionUtils<D>, const D: usize>(
    h: &[FT],
    i: usize,
    params: SimulationParams,
) -> FT {
    smoothing_length_single(h, i, params) * DU::support_radius_by_smoothing_length()
}

/** Determine kernel support radius */
pub fn smoothing_length_fluid_boundary(h: &[FT], i: usize, h_b: FT, params: SimulationParams) -> FT {
    match PARTICLE_SIZES {
        ParticleSizes::Adaptive => (h[i] + h_b) * 0.5,
        ParticleSizes::Uniform => params.h,
    }
}

/** Determine kernel support radius */
pub fn smoothing_length_boundary_boundary(h_bi: FT, h_bj: FT, _params: SimulationParams) -> FT {
    assert!(h_bi == h_bj);
    h_bi
}

// /** Determine kernel support radius */
// pub fn H(h: &[FT], i: usize, j: usize, params: SimulationParams) -> FT {
//     match PARTICLE_SIZES {
//         ParticleSizes::Adaptive => (h[i] + h[j]) * 0.5,
//         ParticleSizes::Uniform => params.kernel_support_radius,
//     }
// }

// /** Determine kernel support radius where j is a boundary particle */
// pub fn Hfb(h: &[FT], i: usize, hj: FT, params: SimulationParams) -> FT {
//     match PARTICLE_SIZES {
//         ParticleSizes::Adaptive => (h[i] + hj) * 0.5,
//         ParticleSizes::Uniform => params.kernel_support_radius,
//     }
// }

// /** Determine kernel support radius where i and j are a boundary particles */
// pub fn Hbb(hi: FT, hj: FT, params: SimulationParams) -> FT {
//     match PARTICLE_SIZES {
//         ParticleSizes::Adaptive => (hi + hj) * 0.5,
//         ParticleSizes::Uniform => params.kernel_support_radius,
//     }
// }
