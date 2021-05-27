use crate::{
    concurrency::par_iter_mut1, floating_type_mod::FT, neighborhood_search::NeighborhoodCache,
    simulation_parameters::SimulationParams, sph_kernels::DimensionUtils, LevelEstimationState, ParticleVec,
};

/**
 * From "Infinite Continuous Adaptivity for Incompressible SPH" Eq. 5.
 *
 * It depends on "ratio = mass / optimal mass"
 */
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum ParticleSizeClass {
    // ratio <= 0.5  => merge into neighbors
    TooSmall,
    // 0.5 < ratio <= 0.9  => receive distributed mass
    Small,
    // 0.9 < ratio < 1.1  => do nothing
    Optimal,
    // 1.1 <= ratio < 2  => distribute mass
    Large,
    // 2 <= ratio  => split
    TooLarge,
}
pub const PARTICLE_SIZE_FACTOR_TOO_SMALL: FT = 0.5;
pub const PARTICLE_SIZE_FACTOR_SMALL: FT = 1. / 1.1;
pub const PARTICLE_SIZE_FACTOR_LARGE: FT = 1.1;
pub const PARTICLE_SIZE_FACTOR_TOO_LARGE: FT = 2.0;

pub const MERGE_PARTNER_AVAILABLE: u32 = 0xFFFFFFFF;
pub const MERGE_PARTNER_DELETE: u32 = 0xFFFFFFFE;

pub fn classify_particle<DU: DimensionUtils<D>, const D: usize>(
    level_estimation: &LevelEstimationState,
    mass: FT,
    simulation_params: SimulationParams,
) -> ParticleSizeClass {
    let optimal_mass = level_estimation.target_mass::<DU, D>(simulation_params);
    let mrel: FT = mass / optimal_mass;
    assert!(mrel > 0.);

    match mrel {
        x if x <= PARTICLE_SIZE_FACTOR_TOO_SMALL => ParticleSizeClass::TooSmall,
        x if x <= PARTICLE_SIZE_FACTOR_SMALL => ParticleSizeClass::Small,
        x if x < PARTICLE_SIZE_FACTOR_LARGE => ParticleSizeClass::Optimal,
        x if x < PARTICLE_SIZE_FACTOR_TOO_LARGE => ParticleSizeClass::Large,
        _ => ParticleSizeClass::TooLarge,
    }
}

pub fn classify_particles<DU: DimensionUtils<D>, const D: usize>(
    particles: &mut ParticleVec<D>,
    _neighs: &NeighborhoodCache,
    simulation_params: SimulationParams,
) {
    par_iter_mut1(&mut particles.particle_size_class, |i, p_particle_size_class| {
        *p_particle_size_class =
            classify_particle::<DU, D>(&particles.level_estimation[i], particles.mass[i], simulation_params);
    });
}

pub mod particle_merging;
pub mod particle_sharing;
pub mod splitting;
