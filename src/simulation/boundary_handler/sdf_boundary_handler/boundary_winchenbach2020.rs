use std::marker::PhantomData;

use crate::{
    boundary_handler::BoundaryHandlerTrait,
    concurrency::par_iter_mut1,
    floating_type_mod::FT,
    neighborhood_search::NeighborhoodCache,
    sdf::Sdf,
    simulation_parameters::{BoundaryPenaltyTerm, OperatorDiscretization, SimulationParams},
    sph_kernels::{smoothing_length, support_radius_single, DimensionUtils},
    VF,
};

use super::{
    plane_numerics::{dlambda, lambda, LFT},
    LookupTable1D,
};
use nalgebra::zero;

/* From paper: Semi-analytic boundary handling below particle resolution for smoothed particle hydrodynamics */
pub struct BoundaryWinchenbach2020<DU: DimensionUtils<D>, const D: usize> {
    pub sdf: Vec<Sdf<D>>,
    lambda_lut: LookupTable1D,
    dlambda_lut: LookupTable1D,

    // lambda and the gradient of lambda (including penalty terms see Eq. 29 and Eq. 30 "Semi-Analytic Boundary Handling Below Particle Resolution for Smoothed Particle Hydrodynamics")
    pub lambda: Vec<Vec<(FT, VF<D>)>>,

    _marker: PhantomData<DU>,
}

impl<DU: DimensionUtils<D>, const D: usize> BoundaryWinchenbach2020<DU, D> {
    pub fn new(sdf: Vec<Sdf<D>>, num_fluid_particles: usize) -> Self {
        let steps = 10000;
        let lambda_lut = LookupTable1D::new(-1., 1., steps, |x: FT| lambda::<D>(x as LFT) as FT);
        let dlambda_lut = LookupTable1D::new(-1., 1., steps, |x: FT| dlambda::<D>(x as LFT) as FT);

        Self {
            sdf,
            lambda_lut,
            dlambda_lut,
            lambda: vec![vec![]; num_fluid_particles],
            _marker: PhantomData::default(),
        }
    }

    pub fn lambda_sum(&self, i: usize) -> FT {
        self.lambda[i].iter().map(|(lambda, _lambda_grad)| lambda).sum()
    }
}

#[inline(always)]
fn rho_b(_rho_0: FT, _rho_i: FT) -> FT {
    _rho_0
}

impl<'sdf, DU: DimensionUtils<D>, const D: usize> BoundaryHandlerTrait<D> for BoundaryWinchenbach2020<DU, D> {
    fn update_after_advect(
        &mut self,
        fluid_particle_positions: &[crate::VF<D>],
        fluid_h: &[FT],
        simulation_params: crate::simulation_parameters::SimulationParams,
    ) {
        par_iter_mut1(&mut self.lambda, |i, p_lambda| {
            let xi = fluid_particle_positions[i];

            // TODO: assert that the kernel function is the cubic kernel
            let sr_i = support_radius_single::<DU, D>(fluid_h, i, simulation_params);

            p_lambda.clear();

            for sdf in &self.sdf {
                let d = sdf.probe(xi) / sr_i;

                if d < 1. {
                    // this particles support volume is not in contact with the boundary
                    // only need to compute the gradient if particle radius intersects with boundary

                    let mut sdf_grad = sdf.finite_diff_gradient(xi, simulation_params.sdf_gradient_eps);
                    let sdf_grad_norm = sdf_grad.norm();
                    if sdf_grad_norm >= 0.00001 {
                        sdf_grad /= sdf_grad_norm;

                        let penalty = match simulation_params.boundary_penalty_term {
                            BoundaryPenaltyTerm::None => 1.,
                            BoundaryPenaltyTerm::Linear => 1. - d,
                            BoundaryPenaltyTerm::Quadratic1 => {
                                if d > 0. {
                                    1.
                                } else if d > -1. {
                                    0.5 * d * d + 1.
                                } else {
                                    0.5 - d
                                }
                            }
                            BoundaryPenaltyTerm::Quadratic2 => {
                                if d > 0. {
                                    1.
                                } else if d > -0.5 {
                                    d * d + 1.
                                } else {
                                    0.75 - d
                                }
                            }
                        };

                        let penalty_derivative: FT = match simulation_params.boundary_penalty_term {
                            BoundaryPenaltyTerm::None => 0.,
                            BoundaryPenaltyTerm::Linear => -1.,
                            BoundaryPenaltyTerm::Quadratic1 => {
                                if d > 0. {
                                    0.
                                } else if d > -1. {
                                    d
                                } else {
                                    -1.
                                }
                            }
                            BoundaryPenaltyTerm::Quadratic2 => {
                                if d > 0. {
                                    0.
                                } else if d > -0.5 {
                                    2. * d
                                } else {
                                    -1.
                                }
                            }
                        };

                        let lambda: FT;
                        let lambda_derivative: FT;
                        if d <= -1. {
                            lambda = 1.;
                            lambda_derivative = 0.;
                        } else {
                            lambda = self.lambda_lut.get(d);
                            lambda_derivative = self.dlambda_lut.get(d);
                        }

                        let lambda_penalty = lambda * penalty;
                        let grad_lambda_penalty =
                            sdf_grad / sr_i * (penalty_derivative * lambda + penalty * lambda_derivative);

                        p_lambda.push((lambda_penalty, grad_lambda_penalty));
                    } else {
                        // the gradient is not well defined
                        // TODO: research what is best option in this case (normalize -> erratic?)
                    }
                }
            }
        });
    }

    fn density_boundary_term(
        &self,
        i: usize,
        _fluid_particle_positions: &[crate::VF<D>],
        _fluid_h: &[FT],
        _simulation_params: crate::simulation_parameters::SimulationParams,
    ) -> FT {
        self.lambda[i].iter().map(|(lambda, _grad_lambda)| lambda).sum()
    }

    fn iisph_boundary_pressure_accel(
        &self,
        i: usize,
        _fluid_particle_positions: &[crate::VF<D>],
        _fluid_particle_masses: &[crate::floating_type_mod::FT],
        fluid_particle_pressure: impl Fn(usize) -> FT,
        fluid_particle_density: &[crate::floating_type_mod::FT],
        _fluid_h: &[FT],
        simulation_params: crate::simulation_parameters::SimulationParams,
    ) -> crate::VF<D> {
        let mut result = VF::<D>::zeros();
        for (_lambda, grad_lambda) in &self.lambda[i] {
            let p_i = fluid_particle_pressure(i);
            let p_ib = match simulation_params.operator_discretization {
                OperatorDiscretization::ConsistentSymmetricGradient => {
                    // TODO: use MLS interpolation instead of pressure mirroring
                    p_i
                }
                OperatorDiscretization::ConsistentSimpleGradient | OperatorDiscretization::Winchenbach2020 => {
                    //
                    0.
                }
            };
            let rho_i = fluid_particle_density[i];
            let rho_b = rho_b(simulation_params.rest_density, rho_i);

            // equation 47 of Winchenbach2020
            result += -rho_b * (p_i / (rho_i * rho_i) + p_ib / (rho_b * rho_b)) * grad_lambda
        }
        result
    }

    fn calculate_divergence_iisph(
        &self,
        i: usize,
        quantity_f: impl Fn(usize) -> VF<D>,
        quantity_b: VF<D>,
        _position: &[VF<D>],
        density: &[FT],
        _fluid_h: &[FT],
        simulation_params: SimulationParams,
    ) -> FT {
        let mut result: FT = 0.;
        for (_lambda, grad_lambda) in &self.lambda[i] {
            let rho_i = density[i];
            let rho_0 = simulation_params.rest_density;
            let rho_b = rho_b(rho_0, rho_i);

            match simulation_params.operator_discretization {
                OperatorDiscretization::ConsistentSimpleGradient
                | OperatorDiscretization::ConsistentSymmetricGradient => {
                    result += rho_b / rho_i * (quantity_b - quantity_f(i)).dot(&grad_lambda);
                }
                OperatorDiscretization::Winchenbach2020 => {
                    result += (quantity_b - quantity_f(i)).dot(&grad_lambda);
                }
            }
        }
        result
    }

    fn iisph_aii(
        &self,
        i: usize,
        fluid_positions: &[VF<D>],
        fluid_masses: &[FT],
        fluid_densities: &[FT],
        fluid_fluid_neighs: &NeighborhoodCache,
        fluid_h: &[FT],
        simulation_params: SimulationParams,
    ) -> FT {
        match simulation_params.operator_discretization {
            OperatorDiscretization::Winchenbach2020 => {
                let mi = fluid_masses[i];
                let rho_i = fluid_densities[i];
                let rho_0 = simulation_params.rest_density;
                let rho_i_sq = rho_i * rho_i;
                let rho_b = rho_b(rho_0, rho_i);

                let mut mj_wij: VF<D> = zero();
                let mut mj_by_rhoj_wij: VF<D> = zero();
                let mut mj_by_rhoj_wij_sq: FT = zero();
                for j in fluid_fluid_neighs.iter(i) {
                    let hij = smoothing_length(fluid_h, i, j, simulation_params);
                    let gw_ij: VF<D> = DU::kernel_derivh(fluid_positions[i] - fluid_positions[j], hij);

                    mj_wij += fluid_masses[j] * gw_ij;
                    mj_by_rhoj_wij += fluid_masses[j] / fluid_densities[j] * gw_ij;
                    mj_by_rhoj_wij_sq += fluid_masses[j] / fluid_densities[j] * gw_ij.norm_squared();
                }

                let mut sum_glambda: VF<D> = zero();
                let mut rhob_glambda: VF<D> = zero();
                let mut sum_boundary: VF<D> = zero();

                for (_lambda, grad_lambda) in &self.lambda[i] {
                    let p_ib_coeff = 0.;

                    rhob_glambda += rho_b * grad_lambda;
                    sum_glambda += grad_lambda;
                    sum_boundary += rho_b * (1. / (rho_i * rho_i) + p_ib_coeff / (rho_b * rho_b)) * grad_lambda;
                }

                (mj_wij / rho_i_sq + sum_boundary).dot(&(mj_by_rhoj_wij + sum_glambda))
                    + (mi * mj_by_rhoj_wij_sq / rho_i_sq)
            }
            OperatorDiscretization::ConsistentSimpleGradient | OperatorDiscretization::ConsistentSymmetricGradient => {
                let mi = fluid_masses[i];
                let rho_i = fluid_densities[i];
                let rho_0 = simulation_params.rest_density;
                let rho_i_sq = rho_i * rho_i;
                let rho_i_cu = rho_i * rho_i * rho_i;

                let mut mj_wij: VF<D> = zero();
                let mut mj_wij_sq: FT = zero();
                for j in fluid_fluid_neighs.iter(i) {
                    let hij = smoothing_length(fluid_h, i, j, simulation_params);
                    let gw_ij: VF<D> = DU::kernel_derivh(fluid_positions[i] - fluid_positions[j], hij);

                    mj_wij += fluid_masses[j] * gw_ij;
                    mj_wij_sq += fluid_masses[j] * gw_ij.norm_squared();
                }

                let mut rhob_glambda: VF<D> = zero();
                let mut sum_boundary: VF<D> = zero();

                for (_lambda, grad_lambda) in &self.lambda[i] {
                    let rho_b = rho_b(rho_0, rho_i);
                    let p_ib_coeff = match simulation_params.operator_discretization {
                        OperatorDiscretization::ConsistentSymmetricGradient
                        | OperatorDiscretization::Winchenbach2020 => 1.,
                        OperatorDiscretization::ConsistentSimpleGradient => 0.,
                    };

                    rhob_glambda += rho_b * grad_lambda;
                    sum_boundary += rho_b * (1. / (rho_i * rho_i) + p_ib_coeff / (rho_b * rho_b)) * grad_lambda;
                }

                (mj_wij / rho_i_sq + sum_boundary).dot(&(mj_wij / rho_i + rhob_glambda / rho_i))
                    + (mi * mj_wij_sq) / rho_i_cu
            }
        }
    }

    fn swap(&mut self, i: usize, j: usize) {
        self.lambda.swap(i, j);
    }

    fn truncate(&mut self, len: usize) {
        self.lambda.truncate(len);
    }

    fn extend(&mut self, num_elements: usize) {
        self.lambda.extend((0..num_elements).map(|_| vec![]));
    }

    fn distance_to_boundary(&self, i: usize, fluid_positions: &[VF<D>], _simulation_params: SimulationParams) -> FT {
        self.sdf
            .iter()
            .map(|sdf| sdf.probe(fluid_positions[i]))
            .fold(FT::INFINITY, |a, b| FT::min(a, b))
    }
}
