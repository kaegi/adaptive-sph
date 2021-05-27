/** Floating type for the lambda functions */
pub type LFT = f64;
use std::f64::consts::PI;

/** Redefinitions so that CAS-generated formulas can simply be copied from Maxima (or Matlab, Mathematica, ...) */
fn pow(x: LFT, i: i32) -> LFT {
    LFT::powi(x, i)
}
fn ln(x: LFT) -> LFT {
    LFT::ln(x)
}
fn acos(x: LFT) -> LFT {
    LFT::acos(x)
}
fn sqrt(x: LFT) -> LFT {
    LFT::sqrt(x)
}

fn lambda2(d: LFT) -> LFT {
    if d >= 0. {
        lambda2_nonnegative(d)
    } else {
        1. - lambda2_nonnegative(-d)
    }
}

/**
 For the cubic spline kernel with support radius h=1.
*/
fn lambda2_nonnegative(d: LFT) -> LFT {
    assert!(d >= 0.);

    /*
    Same derivation as "Semi-Analytic Boundary Handling Below Particle Resolution for
    Smoothed Particle Hydrodynamics" Eq. 57.
    But this was generated using maxima. The file is included in this project.
    */

    if d < 0.000000001 {
        0.5
    } else if d < 0.5 {
        (((-48.0 * pow(d, 5)) - 80.0 * pow(d, 3)) * ln(sqrt(1.0 - 2.0 * d) * sqrt(2.0 * d + 1.0) + 1.0)
            + (12.0 * pow(d, 5) + 80.0 * pow(d, 3)) * ln(sqrt(1.0 - 1.0 * d) * sqrt(d + 1.0) + 1.0)
            - 1.0 * acos(2.0 * d)
            + 36.0 * ln(d) * pow(d, 5)
            + 48.0 * ln(2.0) * pow(d, 5)
            + sqrt(1.0 - 2.0 * d) * sqrt(2.0 * d + 1.0) * (68.0 * pow(d, 3) + 8.0 * d)
            + 80.0 * ln(2.0) * pow(d, 3)
            + sqrt(1.0 - 1.0 * d) * sqrt(d + 1.0) * ((-68.0 * pow(d, 3)) - 32.0 * d)
            + 8.0 * acos(d))
            / (7. * PI)
    } else if d < 1. {
        -(((-12.0 * pow(d, 5)) - 80.0 * pow(d, 3)) * ln(sqrt(1.0 - 1.0 * d) * sqrt(d + 1.0) + 1.0)
            + ln(d) * (12.0 * pow(d, 5) + 80.0 * pow(d, 3))
            + sqrt(1.0 - 1.0 * d) * sqrt(d + 1.0) * (68.0 * pow(d, 3) + 32.0 * d)
            - 8.0 * acos(d))
            / (7. * PI)
    } else {
        0.
    }
}

/*
For the cubic spline kernel with support radius h=1.
*/
fn dlambda2(d: LFT) -> LFT {
    if d >= 0. {
        return dlambda2_nonnegative(d);
    } else {
        return dlambda2_nonnegative(-d);
    }
}

/*
For the cubic spline kernel with support radius h=1.
*/
fn dlambda2_nonnegative(d: LFT) -> LFT {
    assert!(d >= 0.);

    if d < 0.0000000001 {
        -1.36418522650196
    } else if d < 0.5 {
        // this formula undefined for d==0.5 (because of "0 to a negative exponent")
        -(1.0
            * (sqrt(2.0 * d + 1.0)
                * (sqrt(1.0 - 2.0 * d)
                    * ((240.0 * pow(d, 2) - 240.0 * pow(d, 6)) * ln(sqrt(1.0 - 2.0 * d) * sqrt(2.0 * d + 1.0) + 1.0)
                        + (60.0 * pow(d, 6) + 180.0 * pow(d, 4) - 240.0 * pow(d, 2))
                            * ln(sqrt(1.0 - 1.0 * d) * sqrt(d + 1.0) + 1.0)
                        + ln(d) * (180.0 * pow(d, 6) - 180.0 * pow(d, 4))
                        + (240.0 * ln(2.0) - 1040.0) * pow(d, 6)
                        + 1000.0 * pow(d, 4)
                        + (10.0 - 240.0 * ln(2.0)) * pow(d, 2)
                        + 30.0)
                    + sqrt(1.0 - 2.0 * d)
                        * sqrt(1.0 - 1.0 * d)
                        * sqrt(d + 1.0)
                        * ((240.0 * pow(d, 4) + 240.0 * pow(d, 2))
                            * ln(sqrt(1.0 - 2.0 * d) * sqrt(2.0 * d + 1.0) + 1.0)
                            + ((-60.0 * pow(d, 4)) - 240.0 * pow(d, 2))
                                * ln(sqrt(1.0 - 1.0 * d) * sqrt(d + 1.0) + 1.0)
                            - 180.0 * ln(d) * pow(d, 4)
                            + (780.0 - 240.0 * ln(2.0)) * pow(d, 4)
                            - 240.0 * ln(2.0) * pow(d, 2)
                            + 30.0))
                + sqrt(1.0 - 1.0 * d)
                    * sqrt(d + 1.0)
                    * (((-960.0 * pow(d, 6)) - 720.0 * pow(d, 4) + 240.0 * pow(d, 2))
                        * ln(sqrt(1.0 - 2.0 * d) * sqrt(2.0 * d + 1.0) + 1.0)
                        + (240.0 * pow(d, 6) + 900.0 * pow(d, 4) - 240.0 * pow(d, 2))
                            * ln(sqrt(1.0 - 1.0 * d) * sqrt(d + 1.0) + 1.0)
                        + ln(d) * (720.0 * pow(d, 6) - 180.0 * pow(d, 4))
                        + (960.0 * ln(2.0) + 1040.0) * pow(d, 6)
                        + (720.0 * ln(2.0) - 100.0) * pow(d, 4)
                        + ((-240.0 * ln(2.0)) - 160.0) * pow(d, 2)
                        + 30.0)
                + (960.0 * pow(d, 8) - 240.0 * pow(d, 6) - 960.0 * pow(d, 4) + 240.0 * pow(d, 2))
                    * ln(sqrt(1.0 - 2.0 * d) * sqrt(2.0 * d + 1.0) + 1.0)
                + ((-240.0 * pow(d, 8)) - 660.0 * pow(d, 6) + 1140.0 * pow(d, 4) - 240.0 * pow(d, 2))
                    * ln(sqrt(1.0 - 1.0 * d) * sqrt(d + 1.0) + 1.0)
                - 960.0 * ln(2.0) * pow(d, 8)
                + ln(d) * ((-720.0 * pow(d, 8)) + 900.0 * pow(d, 6) - 180.0 * pow(d, 4))
                + 240.0 * ln(2.0) * pow(d, 6)
                + (960.0 * ln(2.0) + 120.0) * pow(d, 4)
                + ((-240.0 * ln(2.0)) - 150.0) * pow(d, 2)
                + 30.0))
            / (28.0 * PI * pow(d, 4)
                + sqrt(2.0 * d + 1.0)
                    * (sqrt(1.0 - 2.0 * d) * (7.0 * PI - 7.0 * PI * pow(d, 2))
                        + 7.0 * PI * sqrt(1.0 - 2.0 * d) * sqrt(1.0 - 1.0 * d) * sqrt(d + 1.0))
                + sqrt(1.0 - 1.0 * d) * sqrt(d + 1.0) * (7.0 * PI - 28.0 * PI * pow(d, 2))
                - 35.0 * PI * pow(d, 2)
                + 7.0 * PI)
    } else if d < 1. {
        (sqrt(1.0 - 1.0 * d)
            * sqrt(d + 1.0)
            * ((60.0 * pow(d, 4) + 240.0 * pow(d, 2)) * ln(sqrt(1.0 - 1.0 * d) * sqrt(d + 1.0) + 1.0)
                + 260.0 * pow(d, 4)
                + ln(d) * ((-60.0 * pow(d, 4)) - 240.0 * pow(d, 2))
                - 220.0 * pow(d, 2)
                - 40.0)
            + ((-60.0 * pow(d, 6)) - 180.0 * pow(d, 4) + 240.0 * pow(d, 2))
                * ln(sqrt(1.0 - 1.0 * d) * sqrt(d + 1.0) + 1.0)
            + ln(d) * (60.0 * pow(d, 6) + 180.0 * pow(d, 4) - 240.0 * pow(d, 2))
            + 260.0 * pow(d, 4)
            - 220.0 * pow(d, 2)
            - 40.0)
            / ((-7.0 * PI * pow(d, 2)) + 7.0 * PI * sqrt(1.0 - 1.0 * d) * sqrt(d + 1.0) + 7.0 * PI)
    } else {
        0.
    }
}

pub fn lambda<const D: usize>(x: LFT) -> LFT {
    if D == 2 {
        lambda2(x)
    } else if D == 3 {
        todo!()
    } else {
        unreachable!()
    }
}

pub fn dlambda<const D: usize>(x: LFT) -> LFT {
    if D == 2 {
        dlambda2(x)
    } else if D == 3 {
        todo!()
    } else {
        unreachable!()
    }
}

#[cfg(test)]
mod tests {

    use super::{dlambda2, lambda2, LFT};
    use crate::{assert_ft_approx_eq2, sph_kernels, floating_type_mod::FT, V};

    #[test]
    fn test_dlambda2_specific_values() {
        let values = [
            (1.0e-5, -1.364185225745495),
            (0.1, -1.291255734976317),
            (0.2, -1.09590958428671),
            (0.3, -0.8294373145386852),
            (0.475, -0.3694455226951835),
            (0.49999999, -0.3172459084022253),
            (0.5, -0.3172458884798477),
            (0.6, -0.1553847490374719),
            (0.7, -0.06022919733948317),
            (0.8, -0.01536108745740005),
            (0.9, -0.001424092559566546),
            (0.9999999999, -1.37123132821062e-10),
        ];

        for (x, y) in values.into_iter().rev() {
            println!("d={}\ncode={}\tmaxima={}", x, dlambda2(x), y);
            assert_ft_approx_eq2(dlambda2(x), y, 0.00000001, || {
                (format!("dlambda2 for d={}", x), format!("Code"), format!("Maxima"))
            });
        }
    }

    #[test]
    fn test_dlambda2_finite_diffs() {
        let steps = 300000;
        let eps: LFT = 0.0000001;
        let tolerance = 0.0000001;

        for i in -steps..=steps {
            let x = i as LFT / steps as LFT;
            let dlambda_numerical = (lambda2(x + eps) - lambda2(x - eps)) / (2. * eps);
            let dlambda_analytical = dlambda2(x);

            assert_ft_approx_eq2(dlambda_analytical, dlambda_numerical, tolerance, || {
                (
                    format!("dlambda for x={}", x),
                    format!("dlambda_analytical"),
                    format!("dlambda_numerical"),
                )
            });
        }
    }

    #[test]
    fn test_lambda2_specific_values() {
        // evaluated these values using MAXIMA
        let values = [
            (1.0e-5, 0.4999863581477375),
            (0.1, 0.3660454031974235),
            (0.2, 0.2458568798927798),
            (0.3, 0.1492433688434099),
            (0.475, 0.04601588929110174),
            (0.5, 0.03744216427059437),
            (0.6, 0.01442031051340694),
            (0.7, 0.00413432923941152),
            (0.8, 6.949615905699156e-4),
            (0.9, 3.190640160164168e-5),
            (1.0, 0.),
        ];

        for (x, y) in values.into_iter().rev() {
            println!("d={}\ncode={}\tmaxima={}", x, lambda2(x), y);
            assert_ft_approx_eq2(lambda2(x), y, 0.00000001, || {
                (format!("lambda2 for d={}", x), format!("Code"), format!("Maxima"))
            });
        }
    }

    #[test]
    fn test_lambda2_integrations() {
        for h in [1., 0.0001, 0.05, 2., 10.] {
            for step in (-50..=50).rev() {
                let d = (step as LFT / 40.) * h;
                test_lambda2_integration(h, d);
            }
        }
    }

    // check whether lambda integrates cubic spline kernel
    #[allow(dead_code)]
    fn test_lambda2_integration(h: LFT, d: LFT) {
        let support_radius = 2. * h;
        let grid_size = 350;
        let square_len = 2. * support_radius / grid_size as LFT;
        let square_area = square_len * square_len;

        let mut integral = 0.;

        for y in 0..grid_size {
            for x in 0..grid_size {
                let integration_point: V<LFT, 2> = [
                    (x as LFT + 0.5) * square_len - support_radius,
                    (y as LFT + 0.5) * square_len - support_radius,
                ]
                .into();

                let top = (y as LFT + 1.0) * square_len - support_radius;
                let bottom = (y as LFT + 0.0) * square_len - support_radius;

                // the virtual boundary is on top (from y values "d to infinity")
                if bottom >= d {
                    integral +=
                        sph_kernels::cubic_kernel_2d(integration_point.norm() as FT, h as FT) as LFT * square_area;
                } else if top > d {
                    // only partially integrate this patch
                    let integration_area = (top - d) / (top - bottom) * square_area;
                    integral +=
                        sph_kernels::cubic_kernel_2d(integration_point.norm() as FT, h as FT) as LFT * integration_area;
                }
            }
        }

        let integral_analytic = lambda2(d / support_radius);
        println!("d={} h={} numeric={} analytic={}", d, h, integral, integral_analytic);
        assert_ft_approx_eq2(integral_analytic, integral, 0.00001, || {
            (format!("lambda2"), format!("analytic"), format!("numerical"))
        });
    }
}
