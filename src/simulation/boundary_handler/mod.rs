use crate::{
    neighborhood_search::NeighborhoodCache, simulation_parameters::SimulationParams, sph_kernels::DimensionUtils, floating_type_mod::FT,
    VF,
};

mod particle_boundary_handler;
mod sdf_boundary_handler;

use enum_dispatch::enum_dispatch;
pub use particle_boundary_handler::ParticleBasedBoundaryHandler;
pub use sdf_boundary_handler::BoundaryWinchenbach2020;

#[enum_dispatch]
#[allow(unused_variables)]
pub trait BoundaryHandlerTrait<const D: usize> {
    fn update_after_advect(
        &mut self,
        fluid_particle_positions: &[VF<D>],
        fluid_h: &[FT],
        simulation_params: SimulationParams,
    ) {
        unimplemented!()
    }

    fn density_boundary_term(
        &self,
        i: usize,
        fluid_particle_positions: &[VF<D>],
        fluid_h: &[FT],
        simulation_params: SimulationParams,
    ) -> FT {
        unimplemented!()
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
        unimplemented!()
    }

    fn distance_to_boundary(&self, i: usize, fluid_positions: &[VF<D>], simulation_params: SimulationParams) -> FT {
        unimplemented!()
    }

    // IISPH Eq. 15
    fn iisph_boundary_pressure_accel(
        &self,
        i: usize,
        fluid_particle_positions: &[VF<D>],
        fluid_particle_masses: &[FT],
        fluid_particle_pressure: impl Fn(usize) -> FT,
        fluid_particle_density: &[FT],
        fluid_h: &[FT],
        simulation_params: SimulationParams,
    ) -> VF<D> {
        unimplemented!()
    }

    #[allow(unused_variables)]
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
        unimplemented!()
    }

    #[allow(unused_variables)]
    fn swap(&mut self, i: usize, j: usize) {
        unimplemented!()
    }

    #[allow(unused_variables)]
    fn truncate(&mut self, len: usize) {
        unimplemented!()
    }

    #[allow(unused_variables)]
    fn extend(&mut self, num_elements: usize) {
        unimplemented!()
    }
}

pub struct NoBoundaryHandler {}
#[allow(unused_variables)]
impl<const D: usize> BoundaryHandlerTrait<D> for NoBoundaryHandler {
    fn update_after_advect(
        &mut self,
        fluid_particle_positions: &[VF<D>],
        fluid_h: &[FT],
        simulation_params: SimulationParams,
    ) {
    }

    fn density_boundary_term(
        &self,
        i: usize,
        fluid_particle_positions: &[VF<D>],
        fluid_h: &[FT],
        simulation_params: SimulationParams,
    ) -> FT {
        0.
    }

    // IISPH Eq. 15
    fn iisph_boundary_pressure_accel(
        &self,
        i: usize,
        fluid_particle_positions: &[VF<D>],
        fluid_particle_masses: &[FT],
        fluid_particle_pressure: impl Fn(usize) -> FT,
        fluid_particle_density: &[FT],
        fluid_h: &[FT],
        simulation_params: SimulationParams,
    ) -> VF<D> {
        VF::<D>::zeros()
    }

    #[allow(unused_variables)]
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
        0.
    }

    #[allow(unused_variables)]
    fn swap(&mut self, i: usize, j: usize) {}

    #[allow(unused_variables)]
    fn truncate(&mut self, len: usize) {}

    #[allow(unused_variables)]
    fn extend(&mut self, num_elements: usize) {}
}

#[enum_dispatch(BoundaryHandlerTrait)]
pub enum BoundaryHandler<DU: DimensionUtils<D>, const D: usize> {
    ParticleBasedBoundaryHandler(ParticleBasedBoundaryHandler<DU, D>),
    BoundaryWinchenbach2020(BoundaryWinchenbach2020<DU, D>),
    NoBoundaryHandler(NoBoundaryHandler),
}
