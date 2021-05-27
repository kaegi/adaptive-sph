use std::{marker::PhantomData, vec::from_elem};

use nalgebra::zero;

use crate::{
    concurrency::par_iter_mut1,
    neighborhood_search::NeighborhoodCache,
    simulation_parameters::{OperatorDiscretization, SimulationParams},
    sph_kernels::{
        smoothing_length, smoothing_length_boundary_boundary, smoothing_length_fluid_boundary, ParticleSizes,
        PARTICLE_SIZES, DimensionUtils,
    },
    floating_type_mod::FT, INIT_PARTICLE_MASS, VF,
};

use super::BoundaryHandlerTrait;

pub struct ParticleBasedBoundaryHandler<DU: DimensionUtils<D>, const D: usize> {
    // boundary particle data
    pub boundary_masses: Vec<FT>, // calculated like Psi_b at the start of IISPH Section 4
    pub boundary_positions: Vec<VF<D>>,

    // this vector has an entry for every boundary particle
    boundary_boundary_neighs: NeighborhoodCache,

    // this vector has an entry for every fluid particle
    pub fluid_boundary_neighs: NeighborhoodCache,

    last_global_smoothing_length: Option<FT>,

    _marker: PhantomData<DU>,
}

impl<DU: DimensionUtils<D>, const D: usize> ParticleBasedBoundaryHandler<DU, D> {
    pub fn new(boundary_particle_positions: Vec<VF<D>>, num_fluid_particles: usize) -> Self {
        let num_boundary_particles = boundary_particle_positions.len();

        Self {
            boundary_masses: from_elem(INIT_PARTICLE_MASS, num_boundary_particles),
            boundary_positions: boundary_particle_positions,
            boundary_boundary_neighs: NeighborhoodCache::new(num_boundary_particles),
            fluid_boundary_neighs: NeighborhoodCache::new(num_fluid_particles),
            last_global_smoothing_length: None,
            _marker: PhantomData::default(),
        }
    }

    fn recompute_pseudo_masses(&mut self, simulation_params: SimulationParams) {
        assert!(PARTICLE_SIZES == ParticleSizes::Uniform);
        self.boundary_boundary_neighs.build_neighborhood_list_grid::<DU, D>(
            &self.boundary_positions,
            &self.boundary_positions,
            simulation_params.h * DU::support_radius_by_smoothing_length(),
        );

        let rest_density = simulation_params.rest_density;
        let boundary_positions = &self.boundary_positions;

        par_iter_mut1(&mut self.boundary_masses, |bi, boundary_mass| {
            let mut number_density: FT = 0.;
            let hbi = simulation_params.h;
            for bj in self.boundary_boundary_neighs.iter(bi) {
                let hbj = simulation_params.h;
                let h_bibj = smoothing_length_boundary_boundary(hbi, hbj, simulation_params);
                let weight = DU::kernelh(boundary_positions[bi] - boundary_positions[bj], h_bibj);
                number_density += weight;
            }

            *boundary_mass = rest_density / number_density;
        });
    }

    pub fn num_boundary_particles(&self) -> usize {
        self.boundary_positions.len()
    }
}

impl<DU: DimensionUtils<D>, const D: usize> BoundaryHandlerTrait<D> for ParticleBasedBoundaryHandler<DU, D> {
    fn update_after_advect(
        &mut self,
        fluid_particle_positions: &[VF<D>],
        _fluid_h: &[FT],
        simulation_params: SimulationParams,
    ) {
        if self.last_global_smoothing_length != Some(simulation_params.h) {
            // only recompute masses if boundary particles, kernel function or kernel support radius changes
            self.recompute_pseudo_masses(simulation_params);
            self.last_global_smoothing_length = Some(simulation_params.h);
        }

        let num_fluid_particles = fluid_particle_positions.len();
        assert!(self.fluid_boundary_neighs.len() == num_fluid_particles);
        match PARTICLE_SIZES {
            ParticleSizes::Adaptive => {
                // can Akinci2012 be combined with adaptive particles?
                unimplemented!()
            }
            ParticleSizes::Uniform => {
                self.fluid_boundary_neighs.build_neighborhood_list_grid::<DU, D>(
                    &self.boundary_positions,
                    fluid_particle_positions,
                    simulation_params.h * DU::support_radius_by_smoothing_length(),
                );
            }
        }
    }

    fn density_boundary_term(
        &self,
        i: usize,
        fluid_particle_positions: &[VF<D>],
        fluid_h: &[FT],
        simulation_params: SimulationParams,
    ) -> FT {
        // part of the equation that sums over boundary particles in IISPH Eq 14.

        let mut boundary_density: FT = 0.;
        for b in self.fluid_boundary_neighs.iter(i) {
            let hb = simulation_params.h;
            let h_ib = smoothing_length_fluid_boundary(fluid_h, i, hb, simulation_params);
            let x_ij = fluid_particle_positions[i] - self.boundary_positions[b];
            let weight = DU::kernelh(x_ij, h_ib);
            boundary_density += self.boundary_masses[b] * weight;
        }

        boundary_density
    }

    // IISPH Eq. 15
    fn iisph_boundary_pressure_accel(
        &self,
        i: usize,
        fluid_particle_positions: &[VF<D>],
        _fluid_particle_masses: &[FT],
        fluid_particle_pressure: impl Fn(usize) -> FT,
        fluid_particle_density: &[FT],
        fluid_h: &[FT],
        simulation_params: SimulationParams,
    ) -> VF<D> {
        let p_i = fluid_particle_pressure(i);
        let rho_i = fluid_particle_density[i];

        let mut accel: VF<D> = zero();
        for b in self.fluid_boundary_neighs.iter(i) {
            let hb = simulation_params.h;
            let h_ib = smoothing_length_fluid_boundary(fluid_h, i, hb, simulation_params);
            let x_ij: VF<D> = fluid_particle_positions[i] - self.boundary_positions[b];
            let weight_grad: VF<D> = DU::kernel_derivh(x_ij, h_ib);

            match simulation_params.operator_discretization {
                OperatorDiscretization::ConsistentSymmetricGradient | OperatorDiscretization::Winchenbach2020 => {
                    let p_b = p_i;
                    let rho_b = simulation_params.rest_density;
                    accel += -self.boundary_masses[b] * (p_i / (rho_i * rho_i) + p_b / (rho_b * rho_b)) * weight_grad;
                }
                OperatorDiscretization::ConsistentSimpleGradient => {
                    accel += -self.boundary_masses[b] * p_i / (rho_i * rho_i) * weight_grad;
                }
            }
        }

        accel
    }

    fn calculate_divergence_iisph(
        &self,
        i: usize,
        quantity_f: impl Fn(usize) -> VF<D>,
        quantity_b: VF<D>,
        position: &[VF<D>],
        density: &[FT],
        fluid_h: &[FT],
        simulation_params: SimulationParams,
    ) -> FT {
        let mut sum: FT = zero();
        let qi = quantity_f(i);
        for b in self.fluid_boundary_neighs.iter(i) {
            let hb = simulation_params.h;
            let h_ib = smoothing_length_fluid_boundary(fluid_h, i, hb, simulation_params);
            let weight_grad: VF<D> = DU::kernel_derivh(position[i] - self.boundary_positions[b], h_ib);
            sum += self.boundary_masses[b] * (qi - quantity_b).dot(&weight_grad);
        }

        -sum / density[i]
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
        let mut mj_wij: VF<D> = zero();
        let mut mj_wij_sq: FT = zero();
        for j in fluid_fluid_neighs.iter(i) {
            let hij = smoothing_length(fluid_h, i, j, simulation_params);
            let gw_ij: VF<D> = DU::kernel_derivh(fluid_positions[i] - fluid_positions[j], hij);

            mj_wij += fluid_masses[j] * gw_ij;
            mj_wij_sq += fluid_masses[j] * gw_ij.norm_squared();
        }

        let mut mb_wib: VF<D> = zero();
        let mut mb_pib_div_rho_b_sq_w_ib: VF<D> = zero();
        for b in self.fluid_boundary_neighs.iter(i) {
            let h_b = simulation_params.h;
            let h_ib = smoothing_length_fluid_boundary(fluid_h, i, h_b, simulation_params);
            let gw_ib: VF<D> = DU::kernel_derivh(fluid_positions[i] - self.boundary_positions[b], h_ib);

            let rho_b = simulation_params.rest_density;
            let p_ib_coeff = match simulation_params.operator_discretization {
                OperatorDiscretization::ConsistentSymmetricGradient | OperatorDiscretization::Winchenbach2020 => 1.,
                OperatorDiscretization::ConsistentSimpleGradient => 0.,
            };

            mb_wib += self.boundary_masses[b] * gw_ib;
            mb_pib_div_rho_b_sq_w_ib += self.boundary_masses[b] * (p_ib_coeff / (rho_b * rho_b)) * gw_ib;
        }

        let mi = fluid_masses[i];
        let rho_i = fluid_densities[i];
        let rho_i_sq = rho_i * rho_i;
        let rho_i_cu = rho_i * rho_i * rho_i;

        (mj_wij / rho_i_sq + mb_wib / rho_i_sq + mb_pib_div_rho_b_sq_w_ib).dot(&(mj_wij + mb_wib)) / rho_i
            + (mi * mj_wij_sq) / rho_i_cu
    }

    fn swap(&mut self, i: usize, j: usize) {
        self.fluid_boundary_neighs.swap(i, j);
    }

    fn truncate(&mut self, len: usize) {
        self.fluid_boundary_neighs.truncate(len);
    }

    fn extend(&mut self, num_elements: usize) {
        self.fluid_boundary_neighs.extend(num_elements);

        // triggers re-update (TODO: is this actually necessary)
        // self.last_global_smoothing_length = None;
    }

    fn distance_to_boundary(&self, i: usize, fluid_positions: &[VF<D>], _simulation_params: SimulationParams) -> FT {
        let mut min_dist_sq = FT::INFINITY;
        for b in self.fluid_boundary_neighs.iter(i) {
            let dist_sq = (fluid_positions[i] - self.boundary_positions[b]).norm_squared();
            if dist_sq < min_dist_sq {
                min_dist_sq = dist_sq;
            }
        }

        if min_dist_sq.is_infinite() {
            return FT::INFINITY;
        }
        min_dist_sq.sqrt()
    }
}
