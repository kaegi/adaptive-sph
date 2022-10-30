use crate::concurrency::{into_par_iter, par_iter_reduce2};
use crate::simulation::adaptivity::{splitting::SplitPatterns, ParticleSizeClass};
use crate::simulation::boundary_handler::{
    BoundaryHandler, BoundaryHandlerTrait, BoundaryWinchenbach2020, ParticleBasedBoundaryHandler,
};
use crate::simulation::concurrency::{par_iter_mut0, par_iter_mut1, par_iter_mut2, par_iter_mut3};
use crate::simulation::neighborhood_search::{build_neighborhood_list, NeighborhoodCache};
use crate::simulation::simulation_parameters::{
    FillStashWith, InitBoundaryHandlerType, OperatorDiscretization, SimulationParams,
};
use crate::simulation::sph_kernels::{DimensionUtils, DimensionUtils2d};
use crate::{
    floating_type_mod::{FT, PI},
    vec2f, V2, VF,
};
use eframe::epaint::ahash::HashMap;
use std::fmt::Write;
use std::path::Path;

#[cfg(not(target_arch = "wasm32"))]
use rayon::iter::ParallelIterator;

use serde::{Deserialize, Serialize};

use num_traits::Float;

use std::{
    collections::HashSet,
    fmt::Display,
    hash::Hash,
    mem,
    sync::atomic::{AtomicBool, AtomicU32},
};

use std::time::{Duration, Instant};

use std::sync::atomic::Ordering;

use nalgebra::zero;

use crate::{
    adaptivity::{
        classify_particles,
        particle_merging::{find_merge_partner_sequential, merge_particles},
        particle_sharing::{find_share_partner_sequential, share_particles},
        splitting::{precalculate_split_pattern, split_particles, SplitPattern},
    },
    sdf::{Sdf, Sdf2D, SdfPlane},
    simulation_parameters::{
        HybridDfsphDensitySourceTerm, LevelEstimationMethod, PressureSolverMethod, SizingFunction,
        SupportLengthEstimation, ViscosityType,
    },
    sph_kernels::{
        cubic_kernel_unnormalized, cubic_kernel_unnormalized_deriv, smoothing_length, smoothing_length_single,
        support_radius, support_radius_single, ParticleSizes, PARTICLE_SIZES,
    },
};

// type V3F = V<f32, 3>;
// type M3F = M<f32, 3>;
// type V3D = V<f64, 3>;
// type M3D = M<f64, 3>;
// type V3I = V<i32, 3>;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
enum PressureSolverResidualType {
    DensityError,
    DivergenceError,
}

impl PressureSolverResidualType {
    fn to_lowercase_string(self) -> &'static str {
        match self {
            PressureSolverResidualType::DensityError => "density",
            PressureSolverResidualType::DivergenceError => "divergence",
        }
    }
}

#[derive(Clone)]
struct Counter<T> {
    values: Vec<T>,
    last_start: Instant,
}
impl<T> Counter<T> {
    fn new() -> Self {
        Counter::<T> {
            last_start: Instant::now(),
            values: Vec::new(),
        }
    }
    fn add_value(&mut self, v: T) {
        self.values.push(v);
    }
}
impl Counter<FT> {
    fn avg(&self) -> FT {
        self.values.iter().cloned().sum::<FT>() / self.values.len() as FT
    }
    fn min(&self) -> FT {
        self.values.iter().cloned().fold(FT::max_value(), |a, b| FT::min(a, b))
    }
    fn max(&self) -> FT {
        self.values.iter().cloned().fold(FT::min_value(), |a, b| FT::max(a, b))
    }
}
impl Counter<Duration> {
    fn new_pcounter() -> Self {
        Counter::<Duration> {
            last_start: Instant::now(),
            values: Vec::new(),
        }
    }

    fn begin(&mut self) {
        self.last_start = Instant::now();
    }

    fn end(&mut self) {
        self.values.push(Instant::now() - self.last_start);
    }

    fn end_add_to_last(&mut self) {
        let duration = Instant::now() - self.last_start;
        *self.values.last_mut().unwrap() += duration;
    }

    fn avg(&self) -> Duration {
        self.values.iter().cloned().sum::<Duration>() / self.values.len() as u32
    }

    fn sum(&self) -> Duration {
        self.values.iter().cloned().sum::<Duration>()
    }
}

struct ValueCounters {
    counters: HashMap<String, Counter<FT>>,
    enabled: bool,
}
impl ValueCounters {
    fn new(enabled: bool) -> ValueCounters {
        ValueCounters {
            counters: HashMap::default(),
            enabled,
        }
    }

    fn add_value(&mut self, id: &str, v: FT) {
        if self.enabled {
            self.counters
                .entry(id.to_string())
                .or_insert_with(|| Counter::<FT>::new())
                .add_value(v);
        }
    }
}

struct PerformanceCounters {
    counters: HashMap<String, Counter<Duration>>,
    enabled: bool,
}
impl PerformanceCounters {
    fn new(enabled: bool) -> PerformanceCounters {
        PerformanceCounters {
            counters: HashMap::default(),
            enabled,
        }
    }

    fn begin(&mut self, _id: &str) {
        if self.enabled {
            self.counters
                .entry(_id.to_string())
                .or_insert_with(|| Counter::<Duration>::new_pcounter())
                .begin();
        }
    }
    fn end(&mut self, _id: &str) {
        if self.enabled {
            self.counters.get_mut(_id).unwrap().end();
        }
    }
    fn end_add_to_last(&mut self, _id: &str) {
        if self.enabled {
            self.counters.get_mut(_id).unwrap().end_add_to_last();
        }
    }
}

/**
 * This lists stores for each particle the particle indices of its neighbors.
 * Each particle has a start and end index which identifies the slice where the
 * data for that particle is stored.
 */

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum LevelEstimationState {
    FluidSurface(FT),
    FluidInterior,
}
impl LevelEstimationState {
    /** Assume that all level estimations have a distance value */
    pub fn level(&self) -> FT {
        match self {
            &LevelEstimationState::FluidInterior => {
                unreachable!()
            }
            &LevelEstimationState::FluidSurface(x) => x,
        }
    }

    pub fn target_mass<DU: DimensionUtils<D>, const D: usize>(&self, simulation_params: SimulationParams) -> FT {
        // compute optimal mass (see "Infinite Continuous Adaptivity for Incompressible SPH" section 5.2)
        let level = FT::max(self.level(), -simulation_params.maximum_surface_distance);
        assert!(level <= 0.);
        let interpolation = level / -simulation_params.maximum_surface_distance;
        assert!(interpolation >= 0.);
        assert!(interpolation <= 1.);
        match simulation_params.sizing_function {
            SizingFunction::Mass => {
                simulation_params.mass_fine::<DU, D>() * (1. - interpolation)
                    + simulation_params.mass_base::<DU, D>() * interpolation
            }
            SizingFunction::Radius => {
                let target_radius = simulation_params.particle_radius_fine * (1. - interpolation)
                    + simulation_params.particle_radius_base * interpolation;
                DU::radius_to_sphere_volume(target_radius) * simulation_params.rest_density
            }
            SizingFunction::Radius2 => {
                let exponent = 1. / D as FT;
                let target_radius = simulation_params.particle_radius_fine * (1. - interpolation.powf(exponent))
                    + simulation_params.particle_radius_base * interpolation.powf(exponent);
                DU::radius_to_sphere_volume(target_radius) * simulation_params.rest_density
            }
        }
    }
}

macro_rules! decl_particle_vec {
    (pub struct $struct_name:ident<const D: usize> { $(pub $field_name:ident: Vec<$field_type:ty> | $default_value:expr),*$(,)?  }) => {
        pub struct $struct_name<const D: usize> {
            $(
                pub $field_name : Vec<$field_type>,
            )*
        }

        impl<const D: usize> $struct_name<D> {
            pub fn swap(&mut self, i: usize, j: usize) {
                $(
                    self.$field_name.swap(i, j);
                )*
            }

            /*fn pop(&mut self) {
                $(
                    self.$field_name.pop();
                )*
            }*/

            pub fn truncate(&mut self, len: usize) {
                $(
                    self.$field_name.truncate(len);
                )*
            }

            pub fn extend(&mut self, num_elements: usize) {
                $(
                    self.$field_name.extend((0..num_elements).map::<$field_type, _>(|_| $default_value));
                )*
            }

            pub fn default(len: usize) -> Self {
                Self {
                    $(
                        $field_name: (0..len).map(|_| $default_value).collect::<Vec<$field_type>>(),
                    )*
                }
            }
        }
    }
}

decl_particle_vec! {
    pub struct ParticleVec<const D: usize> {
        pub mass: Vec<FT> | 0.,
        pub position: Vec<VF<D>> | zero(),
        pub velocity: Vec<VF<D>> | zero(),
        pub velocity_temp: Vec<VF<D>> | zero(),

        pub pressure_accel: Vec<VF<D>> | zero(),

        pub density: Vec<FT> | 0.,

        pub ppe_source_term: Vec<FT> | 0.,

        pub pressure: Vec<FT> | 0.,
        pub pressure_next_iter: Vec<FT> | 0.,

        // IISPH EQ 9.
        pub aii: Vec<FT> | 0.,

        // for debugging
        pub density_error: Vec<FT> | 0.,
        pub density_error2: Vec<FT> | 0.,

        pub omega: Vec<FT> | 0.,

        // Based on Wichenbach2016 "Constrained Neighbor Lists for SPH-based Fluid Simulations"
        pub h2: Vec<FT> | 0.,
        pub h2_next: Vec<FT> | 0.,

        pub level_estimation: Vec<LevelEstimationState> | LevelEstimationState::FluidInterior,
        pub level_estimation_temp: Vec<LevelEstimationState> | LevelEstimationState::FluidInterior,
        pub level_old: Vec<FT> | 0.,

        // for particle splitting/merging
        pub particle_size_class: Vec<ParticleSizeClass> | ParticleSizeClass::Optimal,
        pub merge_partner: Vec<AtomicU32> | AtomicU32::new(0),
        pub merge_counter: Vec<u16> | 0,

        // integrate <1>_i for each particle i
        pub constant_field: Vec<FT> | 0.,

        // "stash" attributes away for visualization
        pub stash: Vec<FT> | 0.,

        pub flag_neighborhood_reduced: Vec<bool> | false,
        pub flag_is_fluid_surface: Vec<bool> | false,
        pub flag_insufficient_neighs: Vec<bool> | false,

        pub neighbor_count: Vec<usize> | 0,
    }
}

#[allow(dead_code)]
pub const MM: FT = 1. / 1000.;
pub const CM: FT = 1. / 100.;
#[allow(dead_code)]
pub const DM: FT = 1. / 10.;
pub const M: FT = 1. / 1.;

// volume = mass / density
pub const INIT_REST_DENSITY: FT = 1.;
pub const INIT_GRID_SPACING: FT = 1.5 * CM; // in meters
pub const INIT_VOLUME_FILL_RATIO: FT = 0.93;
pub const INIT_PARTICLE_VOLUME: FT = INIT_VOLUME_FILL_RATIO * INIT_GRID_SPACING * INIT_GRID_SPACING as FT; // mass of particle = 1 / volume of particle
pub const INIT_PARTICLE_MASS: FT = INIT_PARTICLE_VOLUME * INIT_REST_DENSITY;

#[inline(always)]
#[allow(unused)]
// This function is left here to show the calculation of specific ETAs
fn eta_from_neighbor_number<DU: DimensionUtils<D>, const D: usize>() -> FT {
    let optimal_neighbor_number = match D {
        // Found by trial and error
        2 => 16.,

        // This value was given in "Constrained Neighbor Lists for SPH-based Fluid Simulations"
        3 => 50.,

        _ => unreachable!(),
    };

    let eta = optimal_neighbor_number.powf(1. / D as FT) / DU::support_radius_by_smoothing_length();
    eta
}

/** Achieved by 55 neighbors in 3D */
pub const ETA: FT = 1.9;

#[inline(always)]
pub fn local_smoothing_length_from_volume<DU: DimensionUtils<D>, const D: usize>(volume: FT) -> FT {
    return ETA * DU::sphere_volume_to_radius(volume);
}

#[inline(always)]
pub fn local_smoothing_length_from_mass<DU: DimensionUtils<D>, const D: usize>(mass: FT, rest_density: FT) -> FT {
    let volume = mass / rest_density;
    local_smoothing_length_from_volume::<DU, D>(volume)
}

// pub fn local_support_radius_length_from_mass<DU: DimensionUtils<D>, const D: usize>(mass: FT, rest_density: FT) -> FT {
//     return local_smoothing_length_from_mass::<DU, D>(mass, rest_density) * DU::support_radius_by_smoothing_length();
// }

pub fn optimal_neighbor_number<DU: DimensionUtils<D>, const D: usize>() -> FT {
    (ETA * DU::support_radius_by_smoothing_length()).powi(D as i32)
}

#[inline]
fn assert_vector_non_nan<const D: usize>(v: &VF<D>, name: &str) {
    for d in 0..D {
        assert!(v[d].is_finite(), "Assertion '{}[{}].is_finite()' failed!", name, d);
    }
}

struct PressureSolverStatistics {
    normal_particle_count: usize,

    // particles where the denominator in jacobi iteration would be close to zero
    singular_particle_count: usize,

    // negative pressure particles
    negative_particle_count: usize,

    // the error is only computed for non-singular particles with a non-clamped (=positive) pressure
    avg_error_times_normal_particle_count: FT,

    // the maximum density error is the absolute (positive) value
    max_error: FT,
}

impl PressureSolverStatistics {
    fn zero() -> PressureSolverStatistics {
        PressureSolverStatistics {
            normal_particle_count: 0,
            singular_particle_count: 0,
            negative_particle_count: 0,
            avg_error_times_normal_particle_count: 0.,
            max_error: 0.,
        }
    }

    fn for_singular_particle() -> PressureSolverStatistics {
        PressureSolverStatistics {
            singular_particle_count: 1,
            ..Self::zero()
        }
    }

    fn for_negative_pressure_particle() -> PressureSolverStatistics {
        PressureSolverStatistics {
            negative_particle_count: 1,
            ..Self::zero()
        }
    }

    fn for_normal_particle(density_error: FT) -> PressureSolverStatistics {
        PressureSolverStatistics {
            normal_particle_count: 1,
            avg_error_times_normal_particle_count: density_error,
            max_error: density_error.abs(),
            ..Self::zero()
        }
    }

    fn normal_particle_count(&self) -> usize {
        self.normal_particle_count
    }

    fn combine(self, other: PressureSolverStatistics) -> PressureSolverStatistics {
        PressureSolverStatistics {
            normal_particle_count: self.normal_particle_count + other.normal_particle_count,
            negative_particle_count: self.negative_particle_count + other.negative_particle_count,
            singular_particle_count: self.singular_particle_count + other.singular_particle_count,
            max_error: FT::max(self.max_error, other.max_error),
            avg_error_times_normal_particle_count: self.avg_error_times_normal_particle_count
                + other.avg_error_times_normal_particle_count,
        }
    }

    fn avg_error(&self) -> FT {
        if self.normal_particle_count > 0 {
            self.avg_error_times_normal_particle_count / self.normal_particle_count as FT
        } else {
            FT::NAN
        }
    }
}

pub struct FluidSimulation<DU: DimensionUtils<D>, const D: usize> {
    pub particles: ParticleVec<D>,
    pub neighs: NeighborhoodCache,
    pub boundary_handler: BoundaryHandler<DU, D>,
    pub time: FT,
    pub split_patterns: SplitPatterns<D>,

    pcounters: PerformanceCounters,
    vcounters: ValueCounters,

    step_number: usize,

    _dimension_utils: std::marker::PhantomData<DU>,
}

impl<DU: DimensionUtils<D>, const D: usize> FluidSimulation<DU, D> {
    fn new(
        fluid_particle_positions: Vec<VF<D>>,
        fluid_particle_velocities: Vec<VF<D>>,
        fluid_particle_masses: Vec<FT>,
        boundary_handler: BoundaryHandler<DU, D>,
        split_patterns: SplitPatterns<D>,
        counters_enabled: bool,
    ) -> Self {
        let num_fluid_particles = fluid_particle_positions.len();
        assert!(fluid_particle_velocities.len() == num_fluid_particles);
        assert!(fluid_particle_masses.len() == num_fluid_particles);

        // if PARTICLE_SIZES == ParticleSizes::Uniform {
        //     for i in 0..fluid_particle_masses.len() {
        //         assert!(is_ft_approx_eq(fluid_particle_masses[i], INIT_PARTICLE_MASS, 0.000001));
        //     }
        // }

        let h_init = match PARTICLE_SIZES {
            ParticleSizes::Adaptive => (0..num_fluid_particles)
                .map(|i| local_smoothing_length_from_mass::<DU, D>(fluid_particle_masses[i], INIT_REST_DENSITY))
                .collect(),
            ParticleSizes::Uniform => {
                // value not used during simulation
                vec![zero(); num_fluid_particles]
            }
        };

        let mut particles = ParticleVec::<D>::default(num_fluid_particles);

        particles.position = fluid_particle_positions;
        particles.mass = fluid_particle_masses;
        particles.velocity = fluid_particle_velocities;
        particles.h2_next = h_init;

        FluidSimulation {
            particles,
            boundary_handler,
            neighs: NeighborhoodCache::new(num_fluid_particles),
            _dimension_utils: std::marker::PhantomData,
            time: 0.,
            step_number: 0,
            split_patterns,
            pcounters: PerformanceCounters::new(counters_enabled),
            vcounters: ValueCounters::new(counters_enabled),
        }
    }

    pub fn num_fluid_particles(&self) -> usize {
        self.particles.position.len()
    }

    fn surface_detection_by_empty_angle(
        simulation_params: SimulationParams,
        mass: &[FT],
        position: &[VF<D>],
        h: &[FT],
        level_estimation: &mut Vec<LevelEstimationState>,
        neighs: &NeighborhoodCache,
        boundary_handler: &BoundaryHandler<DU, D>,
        flag_is_fluid_surface: &mut [bool],
        flag_insufficient_neighs: &mut [bool],
    ) {
        // calculate distance to surface
        par_iter_mut3(
            level_estimation,
            flag_is_fluid_surface,
            flag_insufficient_neighs,
            |i, p_level_estimation, p_flag_is_fluid_surface, p_insufficient_neighs| {
                // TODO: is this the right formula for the radius?
                let particle_radius = DU::sphere_volume_to_radius(mass[i] / simulation_params.rest_density);

                let mut normal: VF<D> = zero();
                for j in neighs.iter(i) {
                    let x_ij = position[i] - position[j];
                    let h_ij = smoothing_length(h, i, j, simulation_params);
                    let dg_ij: VF<D> = DU::kernel_derivh(x_ij, h_ij);
                    // normal_acc -= mass[i] / density[i] * dg_ij;
                    normal -= mass[i] / simulation_params.rest_density * dg_ij;
                }
                // at this point still "|normal| != 1"

                // if there is any particle within a deviation of "threshhold_angle" compared to normal,
                // the particle is classified as "fluid interior"
                let threshhold_angle = 50.;
                let threshhold = FT::cos(threshhold_angle * (crate::floating_type_mod::PI / 180.));
                let mut is_fluid_interior;

                *p_insufficient_neighs = false;
                if neighs.neighbor_count(i) < D * 2 - 1 {
                    is_fluid_interior = false;
                    *p_insufficient_neighs = true;
                } else if normal.norm_squared() < 0.00001 {
                    // if fluid interior particle is surrounded by a symmetric neighborhood (for example at the beginning of the simulation),
                    // the normal is not reliable and set to zero previously
                    is_fluid_interior = true;
                } else if !simulation_params.boundary_is_fluid_surface
                    && boundary_handler.distance_to_boundary(i, position, simulation_params) < h[i] * 1.5
                {
                    is_fluid_interior = true;
                } else {
                    is_fluid_interior = false;
                    normal.normalize_mut();
                    for j in neighs.iter(i) {
                        // if the support radius is chosen as adaptive, it might be estimated
                        // too large -> do not regard particles with a certain distance to matter anymore
                        if !Self::is_neighbor_in_level_estimation_range(
                            simulation_params,
                            position,
                            particle_radius,
                            i,
                            j,
                        ) {
                            continue;
                        }

                        let mut xji = position[j] - position[i];
                        xji.unscale_mut(xji.norm() + 0.000001);

                        if xji.dot(&normal) > threshhold {
                            // there is a particle in normal direction -> fluid interior
                            is_fluid_interior = true;
                            break;
                        }
                    }
                }

                if is_fluid_interior {
                    *p_level_estimation = LevelEstimationState::FluidInterior;
                    *p_flag_is_fluid_surface = false;
                } else {
                    // let surface_level = -0.85 * DU::sphere_volume_to_radius(mass[i] / simulation_params.rest_density);
                    let surface_level = 0.0;
                    *p_level_estimation = LevelEstimationState::FluidSurface(surface_level);
                    *p_flag_is_fluid_surface = true;
                }
            },
        );
    }

    /**
     * Level estimation from "Mass preserving multi-scale SPH". Is working badly for scale interfaces.
     */
    #[inline(always)]
    fn surface_detection_by_center_diff(
        simulation_params: SimulationParams,
        mass: &[FT],
        position: &[VF<D>],
        _density: &[FT],
        h: &[FT],
        level_estimation: &mut Vec<LevelEstimationState>,
        _level_estimation_temp: &mut Vec<LevelEstimationState>,
        neighs: &NeighborhoodCache,
        flag_is_fluid_surface: &mut [bool],
    ) {
        // calculate distance to surface
        par_iter_mut2(
            &mut *level_estimation,
            flag_is_fluid_surface,
            |particle_id, p_level_estimation, p_flag_is_fluid_surface| {
                // TODO: is this the right formula for the radius?
                // let particle_radius = DU::sphere_volume_to_radius(mass[particle_id] / simulation_params.rest_density);

                let mut weight_sum = 0.;
                let mut avg_center: VF<D> = zero();
                let mut avg_radius: FT = zero();

                let mut num_neighbors = 0;

                for neigh_particle_id in neighs.iter(particle_id) {
                    // TODO: is this the right formula for the radius?
                    let neigh_particle_volume = mass[neigh_particle_id] / simulation_params.rest_density;
                    let neigh_particle_radius = DU::sphere_volume_to_radius(neigh_particle_volume);

                    let h_ij = smoothing_length(h, particle_id, neigh_particle_id, simulation_params);

                    let weight =
                        DU::kernelh(position[particle_id] - position[neigh_particle_id], h_ij) * neigh_particle_volume;
                    avg_center += position[neigh_particle_id] * weight;
                    avg_radius += neigh_particle_radius * weight;
                    weight_sum += weight;
                    // avg_radius = FT::max(avg_radius, neigh_particle_radius);

                    num_neighbors = num_neighbors + 1;
                }

                avg_radius /= weight_sum;
                let surface_level = -0.85 * avg_radius;

                let phi;
                if num_neighbors < 5 {
                    phi = surface_level;
                } else {
                    avg_center /= weight_sum;

                    let phi_initial = (position[particle_id] - avg_center).norm() - avg_radius;
                    phi = phi_initial;
                }

                if phi >= surface_level {
                    *p_level_estimation = LevelEstimationState::FluidSurface(phi);
                    *p_flag_is_fluid_surface = true;
                } else {
                    *p_level_estimation = LevelEstimationState::FluidInterior;
                    *p_flag_is_fluid_surface = false;
                }
            },
        );
    }

    #[inline(always)]
    fn is_neighbor_in_level_estimation_range(
        simulation_params: SimulationParams,
        position: &[VF<D>],
        particle_radius: FT,
        i: usize,
        j: usize,
    ) -> bool {
        match simulation_params.support_length_estimation {
            SupportLengthEstimation::FromDistribution | SupportLengthEstimation::FromDistribution2 => {
                // if the support radius is chosen as adaptive, it might be estimated
                // too large -> do not regard particles with a certain distance to matter anymore
                let xji = position[j] - position[i];
                if xji.norm_squared()
                    > (particle_radius * simulation_params.maximum_range)
                        * (particle_radius * simulation_params.maximum_range)
                {
                    return false;
                }

                return true;
            }
            _ => {
                return true;
            }
        }
    }

    /**
     * Level estimation from "Mass preserving multi-scale SPH". Is working badly for scale interfaces.
     */
    #[inline(always)]
    fn propagate_level_set_from_surface_detection(
        simulation_params: SimulationParams,
        position: &[VF<D>],
        mass: &[FT],
        stash: &mut [FT],
        level_estimation: &mut Vec<LevelEstimationState>,
        level_estimation_temp: &mut Vec<LevelEstimationState>,
        neighs: &NeighborhoodCache,
    ) {
        // TODO: this can be avoided and integrated before the first return in the loop
        *level_estimation_temp = level_estimation.clone();

        let mut num_iter = 0;
        let level_estimation_changed = AtomicBool::new(true);
        while level_estimation_changed.load(Ordering::Relaxed) {
            level_estimation_changed.store(false, Ordering::Relaxed);

            par_iter_mut1(level_estimation_temp, |i, p_level_estimation_temp| {
                // TODO: also allow change of NearFluidParticles?
                if let LevelEstimationState::FluidSurface(_) = level_estimation[i] {
                    *p_level_estimation_temp = level_estimation[i];
                    return;
                }

                let particle_radius = DU::sphere_volume_to_radius(mass[i] / simulation_params.rest_density);

                // which is the closes distance to surface we can derive from the neighbors? (the distance is negative
                // because we are inside the fluid)
                let mut maximum_level_estimate_opt: Option<FT> = None;

                for j in neighs.iter(i) {
                    if let LevelEstimationState::FluidSurface(level) = level_estimation[j] {
                        if !Self::is_neighbor_in_level_estimation_range(
                            simulation_params,
                            position,
                            particle_radius,
                            i,
                            j,
                        ) {
                            continue;
                        }

                        let this_level_estimate = level - (position[j] - position[i]).norm();
                        if let Some(maximum_level_estimate) = &mut maximum_level_estimate_opt {
                            *maximum_level_estimate = FT::max(*maximum_level_estimate, this_level_estimate);
                        } else {
                            maximum_level_estimate_opt = Some(this_level_estimate);
                        }
                    }
                }

                if let Some(maximum_level_estimate) = maximum_level_estimate_opt {
                    *p_level_estimation_temp = LevelEstimationState::FluidSurface(maximum_level_estimate);
                    level_estimation_changed.store(true, Ordering::Relaxed);
                } else {
                    *p_level_estimation_temp = LevelEstimationState::FluidInterior;
                }
            });

            mem::swap(level_estimation, level_estimation_temp);

            num_iter = num_iter + 1;

            if simulation_params.fill_stash_with == Some(FillStashWith::SurfaceDistanceMiddle) && num_iter == 1 {
                for i in 0..stash.len() {
                    stash[i] = match level_estimation[i] {
                        LevelEstimationState::FluidInterior => -simulation_params.maximum_surface_distance,
                        LevelEstimationState::FluidSurface(x) => x,
                    };
                }
            }
        }
    }

    #[inline(always)]
    fn smooth_level_estimation_field(
        simulation_params: SimulationParams,
        mass: &[FT],
        position: &[VF<D>],
        density: &[FT],
        h: &[FT],
        // dt: FT,
        level_estimation: &mut Vec<LevelEstimationState>,
        level_estimation_temp: &mut Vec<LevelEstimationState>,
        level_old: &mut Vec<FT>,
        neighs: &NeighborhoodCache,
    ) {
        if simulation_params.level_estimation_method == LevelEstimationMethod::None {
            return;
        }

        // smooth level field (see "Infinite Continuous Adaptivity for Incompressible SPH" section 5.1)
        par_iter_mut2(
            level_estimation_temp,
            level_old,
            |i, p_level_estimation_temp, p_level_old| {
                let mut level = 0.;
                let mut weight = 0.;
                for j in neighs.iter(i) {
                    let x_ij = position[i] - position[j];
                    let h_ij = smoothing_length(&h, i, j, simulation_params);
                    let w_ij = DU::kernelh(x_ij, h_ij);
                    let dist = match level_estimation[j] {
                        LevelEstimationState::FluidInterior => -simulation_params.maximum_surface_distance,
                        LevelEstimationState::FluidSurface(dist) => {
                            FT::max(dist, -simulation_params.maximum_surface_distance)
                        }
                    };
                    assert!(density[j].is_finite());
                    assert!(density[j] > 0.);
                    level += dist * mass[j] / density[j] * w_ij;
                    weight += mass[j] / density[j] * w_ij;
                }

                if !weight.is_finite() || weight <= 0. {
                    panic!("weight is {}<=0 num_neighs:{}", weight, neighs.len());
                }
                level /= weight;
                assert!(level.is_finite());

                // let r = DU::sphere_volume_to_radius(mass[i] / simulation_params.rest_density);
                // level = FT::clamp(level, *p_level_old - r * 1000. * dt, *p_level_old + r * 1000. * dt);
                *p_level_old = level;
                *p_level_estimation_temp = LevelEstimationState::FluidSurface(level);
            },
        );

        mem::swap(level_estimation, level_estimation_temp);
    }

    /**
     * Level estimation from "Mass preserving multi-scale SPH". Is working badly for scale interfaces.
     */
    #[inline(always)]
    fn perform_level_estimation(
        simulation_params: SimulationParams,
        mass: &[FT],
        position: &[VF<D>],
        density: &[FT],
        h: &[FT],
        stash: &mut [FT],
        level_estimation: &mut Vec<LevelEstimationState>,
        level_estimation_temp: &mut Vec<LevelEstimationState>,
        neighs: &NeighborhoodCache,
        boundary_handler: &BoundaryHandler<DU, D>,
        flag_is_fluid_surface: &mut [bool],
        flag_insufficient_neighs: &mut [bool],
    ) {
        match simulation_params.level_estimation_method {
            LevelEstimationMethod::None => {
                return;
            }
            LevelEstimationMethod::CenterDiff => {
                Self::surface_detection_by_center_diff(
                    simulation_params,
                    mass,
                    position,
                    density,
                    h,
                    level_estimation,
                    level_estimation_temp,
                    neighs,
                    flag_is_fluid_surface,
                );
            }
            LevelEstimationMethod::EmptyAngle => Self::surface_detection_by_empty_angle(
                simulation_params,
                mass,
                position,
                h,
                level_estimation,
                neighs,
                boundary_handler,
                flag_is_fluid_surface,
                flag_insufficient_neighs,
            ),
        }

        // TODO: limit change compared to previous iteration

        if simulation_params.fill_stash_with == Some(FillStashWith::SurfaceDistanceFirstIteration) {
            for i in 0..stash.len() {
                stash[i] = match level_estimation[i] {
                    LevelEstimationState::FluidInterior => -simulation_params.maximum_surface_distance,
                    LevelEstimationState::FluidSurface(x) => x,
                };
            }
        }

        Self::propagate_level_set_from_surface_detection(
            simulation_params,
            position,
            mass,
            stash,
            level_estimation,
            level_estimation_temp,
            neighs,
        );
    }

    #[inline(always)]
    // only considers non-pressure forces
    fn calculate_particle_non_pressure_accel(
        i: usize,
        position: &[VF<D>],
        velocity: &[VF<D>],
        density: &[FT],
        mass: &[FT],
        h: &[FT],
        neighs: &NeighborhoodCache,
        simulation_params: SimulationParams,
    ) -> VF<D> {
        let speed_of_sound = 88.; // WCSPH (below eq. 9)

        let mut viscosity_accel = VF::<D>::zeros();

        match simulation_params.viscosity_type {
            ViscosityType::WCSPH => {
                for j in neighs.iter(i) {
                    let x_ab = position[i] - position[j];
                    let v_ab = velocity[i] - velocity[j];

                    let h_ij = smoothing_length(h, i, j, simulation_params);
                    let dg_ij = DU::kernel_derivh(x_ab, h_ij);

                    // compute artifical viscosity acceleration
                    // from WCSPH paper (eq 10)
                    let viscosity_divergence_estimate = v_ab.dot(&x_ab);
                    if viscosity_divergence_estimate < 0. {
                        let viscous_term: FT =
                            2. * simulation_params.viscosity * h_ij * speed_of_sound / (density[i] + density[j]);
                        let pi_ab =
                            -viscous_term * viscosity_divergence_estimate / (x_ab.norm_squared() + 0.001 * h_ij * h_ij);
                        viscosity_accel += -mass[j] * pi_ab * dg_ij;
                        // from WCSPH paper (eq 11)
                    }
                }
            }
            ViscosityType::ApproxLaplace => {
                // panic!("do not use - it's wrong!");
                for j in neighs.iter(i) {
                    let x_ab = position[i] - position[j];
                    let v_ab = velocity[i] - velocity[j];

                    if x_ab.dot(&v_ab) >= 0. {
                        continue;
                    }

                    let h_ij = smoothing_length(h, i, j, simulation_params);
                    let dg_ij = DU::kernel_derivh(x_ab, h_ij);

                    let rho_ij = (density[i] + density[j]) * 0.5;

                    // SPH Tutorial Eq. 102
                    let coeff = 2. * (D + 2) as FT * (mass[j] / rho_ij) * x_ab.dot(&v_ab)
                        / (x_ab.norm_squared() + 0.01 * h_ij * h_ij);

                    viscosity_accel += simulation_params.viscosity * coeff * dg_ij;
                    assert_vector_non_nan(&viscosity_accel, "viscosity_accel");
                }
            }
            ViscosityType::XSPH => {
                // nothing to do here
            }
        }

        assert_vector_non_nan(&viscosity_accel, "viscosity_accel");

        let pull_fluid_gravity: VF<D> = match simulation_params.pull_fluid_to {
            Some(pull_fluid_to) => {
                (VF::<D>::from_iterator(pull_fluid_to.into_iter().cloned()) - position[i]).normalize() * 13.
            }
            None => VF::<D>::zeros(),
        };

        viscosity_accel + simulation_params.gravity_vector() + pull_fluid_gravity
    }

    fn calculate_particle_density(
        i: usize,
        position: &[VF<D>],
        mass: &[FT],
        h: &[FT],
        boundary_handler: &BoundaryHandler<DU, D>,
        neighs: &NeighborhoodCache,
        simulation_params: SimulationParams,
    ) -> FT {
        // calculate density
        let mut density_acc = 0.;
        for j in neighs.iter(i) {
            let x_ab = position[i] - position[j];
            let h_ij = smoothing_length(h, i, j, simulation_params);
            let weight = DU::kernelh(x_ab, h_ij);
            density_acc += mass[j] * weight;
        }

        density_acc += boundary_handler.density_boundary_term(i, position, h, simulation_params);

        density_acc
    }

    fn calculate_all_particle_densities(
        particles: &mut ParticleVec<D>,
        simulation_params: SimulationParams,
        boundary_handler: &BoundaryHandler<DU, D>,
        neighs: &NeighborhoodCache,
    ) {
        par_iter_mut1(&mut particles.density, |i, p_density| {
            *p_density = Self::calculate_particle_density(
                i,
                &particles.position,
                &particles.mass,
                &particles.h2,
                boundary_handler,
                neighs,
                simulation_params,
            );
            assert!(p_density.is_finite());
            assert!(*p_density > 0.0001);
        });
    }

    fn update_velocity_with_non_pressure_accel(
        velocity: &mut Vec<VF<D>>,
        velocity_temp: &mut Vec<VF<D>>,
        density: &[FT],
        position: &[VF<D>],
        mass: &[FT],
        h: &[FT],
        neighs: &NeighborhoodCache,
        simulation_params: SimulationParams,
        dt: FT,
    ) {
        par_iter_mut1(velocity_temp, |i, p_velocity_temp| {
            *p_velocity_temp = velocity[i]
                + dt * Self::calculate_particle_non_pressure_accel(
                    i,
                    position,
                    velocity,
                    density,
                    mass,
                    h,
                    neighs,
                    simulation_params,
                );
        });

        mem::swap(velocity, velocity_temp);
    }

    #[inline(always)]
    fn compute_aii(
        aii: &mut [FT],
        density: &[FT],
        position: &[VF<D>],
        mass: &[FT],
        h: &[FT],
        neighs: &NeighborhoodCache,
        boundary_handler: &BoundaryHandler<DU, D>,
        simulation_params: SimulationParams,
    ) {
        // pre-calculate IISPH parameters for pressure solve iterations
        par_iter_mut1(aii, |i, p_aii| {
            // calculate isph_a_ii
            *p_aii = boundary_handler.iisph_aii(i, position, mass, density, neighs, h, simulation_params);

            // *p_aii = Self::calculate_aii_inefficiently(
            //     i,
            //     density,
            //     position,
            //     mass,
            //     h,
            //     neighs,
            //     boundary_handler,
            //     simulation_params,
            // );

            assert!((*p_aii).is_finite());
        });

        if simulation_params.check_aii {
            par_iter_mut0(position.len(), |check_i| {
                Self::check_aii(
                    check_i,
                    density,
                    aii,
                    position,
                    mass,
                    h,
                    neighs,
                    boundary_handler,
                    simulation_params,
                );
            });
            println!("AII checked: okay!")
        }
    }

    #[inline(always)]
    fn prepare_full_ppe(
        pressure: &mut [FT],
        ppe_source_term: &mut [FT],
        density: &[FT],
        position: &[VF<D>],
        velocity: &[VF<D>],
        mass: &[FT],
        h: &[FT],
        neighs: &NeighborhoodCache,
        boundary_handler: &BoundaryHandler<DU, D>,
        simulation_params: SimulationParams,
        dt: FT,
    ) {
        // pre-calculate IISPH parameters for pressure solve iterations
        par_iter_mut2(pressure, ppe_source_term, |i, p_pressure, p_ppe_source_term| {
            *p_pressure = 0.;
            *p_ppe_source_term = Self::calculate_source_term_full(
                i,
                density,
                velocity,
                position,
                mass,
                h,
                neighs,
                boundary_handler,
                simulation_params,
                dt,
            );
        });
    }

    #[inline(always)]
    fn prepare_only_density_part_ppe(
        pressure: &mut [FT],
        ppe_source_term: &mut [FT],
        density: &[FT],
        simulation_params: SimulationParams,
        dt: FT,
    ) {
        // pre-calculate IISPH parameters for pressure solve iterations
        par_iter_mut2(pressure, ppe_source_term, |i, p_pressure, p_ppe_source_term| {
            *p_pressure = 0.;
            *p_ppe_source_term = Self::calculate_source_term_only_density_part(i, density, simulation_params, dt);
        });
    }

    #[inline(always)]
    fn prepare_ppe_divergence(
        pressure: &mut [FT],
        ppe_source_term: &mut [FT],
        density: &[FT],
        position: &[VF<D>],
        velocity: &[VF<D>],
        mass: &[FT],
        h: &[FT],
        neighs: &NeighborhoodCache,
        boundary_handler: &BoundaryHandler<DU, D>,
        simulation_params: SimulationParams,
        dt: FT,
    ) {
        // pre-calculate IISPH parameters for pressure solve iterations
        par_iter_mut2(pressure, ppe_source_term, |i, p_pressure, p_ppe_source_term| {
            *p_pressure = 0.;
            *p_ppe_source_term = Self::calculate_source_term_divergence(
                i,
                density,
                velocity,
                position,
                mass,
                h,
                neighs,
                boundary_handler,
                simulation_params,
                dt,
            );
        });
    }

    #[inline(always)]
    fn iisph_single_pressure_iteration(
        pressure: &[FT],
        pressure_accel: &mut [VF<D>],
        pressure_next_iter: &mut Vec<FT>,
        density_errors: &mut [FT],
        _level_estimation: &[LevelEstimationState],
        aii: &Vec<FT>,
        density: &[FT],
        ppe_source_term: &[FT],
        position: &[VF<D>],
        mass: &[FT],
        h: &[FT],
        neighs: &NeighborhoodCache,
        boundary_handler: &BoundaryHandler<DU, D>,
        clamp_negative_pressures: bool,
        pressure_solver_residual_type: PressureSolverResidualType,
        simulation_params: SimulationParams,
        dt: FT,
    ) -> PressureSolverStatistics {
        let w = simulation_params.jacobi_omega;

        Self::calculate_particle_pressure_accels(
            pressure_accel,
            position,
            mass,
            pressure,
            density,
            h,
            neighs,
            boundary_handler,
            simulation_params,
        );

        // use relaxed jacobi to update pressure value
        let statistics = par_iter_reduce2(
            pressure_next_iter,
            density_errors,
            PressureSolverStatistics::zero,
            PressureSolverStatistics::combine,
            |i, p_pressure_next_iter, p_density_error| {
                if aii[i].abs() < 10e-4 {
                    // Isolated particles (or particles neighborhoods which are coincident) leads to "aii == 0". To avoid a division by zero when
                    // calculating the pressure, we just set the pressure to zero
                    *p_pressure_next_iter = 0.;
                    return PressureSolverStatistics::for_singular_particle();
                }

                let a_p = Self::calculate_divergence_iisph(
                    i,
                    |j| pressure_accel[j],
                    zero(),
                    position,
                    mass,
                    density,
                    h,
                    neighs,
                    boundary_handler,
                    simulation_params,
                );

                let source_term = ppe_source_term[i];

                if !a_p.is_finite() {
                    panic!("'!a_p.is_finite()' failed. Pressure values probably have exploded!");
                }

                assert!(source_term.is_finite());
                assert!(pressure[i].is_finite());
                assert!(aii[i].is_finite());

                *p_pressure_next_iter = pressure[i] + w * (source_term - a_p) / (aii[i]);

                if !p_pressure_next_iter.is_finite() {
                    panic!("'!p_pressure_next_iter.is_finite()' failed.\n\npressure[i]={}\nw={}\nsource_term={}\na_p={}\naii[i]={}\n", pressure[i], w, source_term, a_p, aii[i]);
                }

                // *p_pressure_next_iter = (*p_pressure_next_iter).min(1000.);
                // if *p_pressure_next_iter > 5000000000. {
                //     panic!("pressure for particle i={} too high:\n\nnext_pressure[i]={}\npressure[i]={}\nw={}\nsource_term={}\na_p={}\naii[i]={}\ndensity[i]={}\n", i, *p_pressure_next_iter, pressure[i], w, source_term, a_p, aii[i], density[i]);
                // }

                let predicted_error: FT;
                match pressure_solver_residual_type {
                    PressureSolverResidualType::DensityError => {
                        // this value is equal to "density(t + Delta t) - rest_density"
                        //
                        // this value will be positive, if the predicted density is over the rest density
                        let predicted_density_error = density[i] * dt * dt * (source_term - a_p);
                        *p_density_error = predicted_density_error;

                        predicted_error = predicted_density_error;
                    }
                    PressureSolverResidualType::DivergenceError => {
                        let predicted_divergence_error = dt * (source_term - a_p);
                        // *p_density_error = predicted_density_error;

                        predicted_error = predicted_divergence_error;
                    }
                }

                // let level = match level_estimation[i] {
                //     LevelEstimationState::FluidSurface(l) => l,
                //     LevelEstimationState::FluidInterior => -100.
                // };

                if *p_pressure_next_iter <= 0. && clamp_negative_pressures {
                    *p_pressure_next_iter = 0.;
                    return PressureSolverStatistics::for_negative_pressure_particle();
                } else {
                    return PressureSolverStatistics::for_normal_particle(predicted_error);
                }
            },
        );

        statistics
    }

    fn calculate_aii_inefficiently(
        i: usize,
        density: &[FT],
        position: &[VF<D>],
        mass: &[FT],
        h: &[FT],
        neighs: &NeighborhoodCache,
        boundary_handler: &BoundaryHandler<DU, D>,
        simulation_params: SimulationParams,
    ) -> FT {
        Self::calculate_pressure_accel_divergence_from_pressures_inefficiently(
            i,
            density,
            position,
            mass,
            |j| if j == i { 1. } else { 0. },
            h,
            neighs,
            boundary_handler,
            simulation_params,
        )
    }

    fn check_aii(
        check_i: usize,
        density: &[FT],
        aii: &[FT],
        position: &[VF<D>],
        mass: &[FT],
        h: &[FT],
        neighs: &NeighborhoodCache,
        boundary_handler: &BoundaryHandler<DU, D>,
        simulation_params: SimulationParams,
    ) {
        let real_a_ii = Self::calculate_aii_inefficiently(
            check_i,
            density,
            position,
            mass,
            h,
            neighs,
            boundary_handler,
            simulation_params,
        );

        #[cfg(feature = "double-precision")]
        let tolerance = 0.00001;
        #[cfg(not(feature = "double-precision"))]
        let tolerance = 0.01;

        assert_ft_approx_eq(aii[check_i], real_a_ii, tolerance, || format!("a_ii[{}]", check_i));
    }

    #[inline]
    fn iisph_pressure_iterations(
        particles: &mut ParticleVec<D>,
        neighs: &NeighborhoodCache,
        boundary_handler: &BoundaryHandler<DU, D>,
        max_avg_error: FT,
        pressure_solver_residual_type: PressureSolverResidualType,
        clamp_negative_pressures: bool,
        simulation_params: SimulationParams,
        dt: FT,
    ) -> (usize, bool) {
        let mut num_pressure_iters = 0;

        for i in 0..particles.position.len() {
            if particles.aii[i] < 0. {
                // panic!(
                //     "AII should not be negative! i={} aii[i]={} ppe_source_term={}",
                //     i, particles.aii[i], particles.ppe_source_term[i]
                // );

                panic!(
                    "AII should not be negative! i={} aii[i]={} ppe_source_term={} pos={:?}",
                    i, particles.aii[i], particles.ppe_source_term[i], particles.position[i]
                );
                // std::thread::sleep(Duration::from_millis(2000));
            }
        }

        let is_converged = loop {
            let statistics = Self::iisph_single_pressure_iteration(
                &particles.pressure,
                &mut particles.pressure_accel,
                &mut particles.pressure_next_iter,
                &mut particles.density_error,
                &particles.level_estimation,
                &particles.aii,
                &particles.density,
                &particles.ppe_source_term,
                &particles.position,
                &particles.mass,
                &particles.h2,
                neighs,
                boundary_handler,
                clamp_negative_pressures,
                pressure_solver_residual_type,
                simulation_params,
                dt,
            );

            mem::swap(&mut particles.pressure, &mut particles.pressure_next_iter);

            if true {
                let max_pressure = if false {
                    particles
                        .pressure
                        .iter()
                        .cloned()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .unwrap()
                } else {
                    -1.
                };

                println!(
                    "Iter {}: avg {} {:.2}% off [max {:.2}% off] ({} normal particles, {} singular particles, {} negative pressure particles) max_pressure: {:.2}",
                    num_pressure_iters,
                    pressure_solver_residual_type.to_lowercase_string(),
                    statistics.avg_error() / simulation_params.rest_density * 100.,
                    statistics.max_error / simulation_params.rest_density * 100.,
                    statistics.normal_particle_count,
                    statistics.singular_particle_count,
                    statistics.negative_particle_count,
                    max_pressure
                );
            }

            match pressure_solver_residual_type {
                PressureSolverResidualType::DensityError => {
                    if statistics.normal_particle_count() == 0
                        || ((statistics.avg_error() / simulation_params.rest_density).abs() < max_avg_error /*0.002*/
                            && num_pressure_iters > 1)
                    {
                        break true;
                    }
                }
                PressureSolverResidualType::DivergenceError => {
                    if statistics.normal_particle_count() == 0
                        || statistics.avg_error().abs() < max_avg_error /*0.0005*/ / dt && num_pressure_iters > 1
                    {
                        break true;
                    }
                }
            }

            // break false;

            if num_pressure_iters == simulation_params.max_iters {
                // panic!("Pressure solver not converged");
                println!("Pressure sover not converged");
                break true;
            }

            num_pressure_iters += 1;
        };

        // Self::calculate_density_errors_simple(
        //     &mut particles.density_error2,
        //     &particles.pressure_next_iter,
        //     &particles.density,
        //     &particles.density_adv,
        //     &particles.position,
        //     &particles.velocity,
        //     &particles.mass,
        //     neighborhood_lists,
        //     &particles.neighborhood_list_indices,
        //     boundary_handler,
        //     simulation_params,
        //     dt,
        // );

        println!("Number of IISPH iterations: {}", num_pressure_iters);

        Self::calculate_particle_pressure_accels(
            &mut particles.pressure_accel,
            &particles.position,
            &particles.mass,
            &particles.pressure,
            &particles.density,
            &particles.h2,
            &neighs,
            boundary_handler,
            simulation_params,
        );

        // if has_negative_aii {
        //     panic!("NEGATIVE A_II");
        // }

        (num_pressure_iters, is_converged)
    }

    #[inline]
    fn calculate_particle_pressure_accels(
        pressure_accel: &mut [VF<D>],
        position: &[VF<D>],
        mass: &[FT],
        pressure: &[FT],
        density: &[FT],
        h: &[FT],
        neighs: &NeighborhoodCache,
        boundary_handler: &BoundaryHandler<DU, D>,
        simulation_params: SimulationParams,
    ) {
        par_iter_mut1(pressure_accel, |i, p_pressure_accel| {
            *p_pressure_accel = Self::calculate_particle_pressure_accel(
                i,
                position,
                mass,
                |i| pressure[i],
                density,
                h,
                neighs,
                boundary_handler,
                simulation_params,
            );
        });
    }

    /**
     * Calculate divergence like in the IISPH paper
     *
     * div(A)_i = -(SUM_j m_j (A_i - A_j) * W_ij) / rho_i
     *
     * which corresponds to Price12 Eq. 79 for \phi=\rho.
     */
    fn calculate_divergence_iisph(
        i: usize,
        quantity_f: impl Fn(usize) -> VF<D>,
        quantity_b: VF<D>,
        position: &[VF<D>],
        mass: &[FT],
        density: &[FT],
        h: &[FT],
        neighs: &NeighborhoodCache,
        boundary_handler: &BoundaryHandler<DU, D>,
        simulation_params: SimulationParams,
    ) -> FT {
        let mut sum: FT = zero();
        let qi = quantity_f(i);

        for j in neighs.iter(i) {
            let h_ij = smoothing_length(h, i, j, simulation_params);
            let dg_ij: VF<D> = DU::kernel_derivh(position[i] - position[j], h_ij);

            match simulation_params.operator_discretization {
                OperatorDiscretization::ConsistentSimpleGradient
                | OperatorDiscretization::ConsistentSymmetricGradient => {
                    sum += mass[j] / density[i] * (quantity_f(j) - qi).dot(&dg_ij);
                }
                OperatorDiscretization::Winchenbach2020 => {
                    sum += mass[j] / density[j] * (quantity_f(j) - qi).dot(&dg_ij);
                }
            }
        }
        let boundary_divergence = boundary_handler.calculate_divergence_iisph(
            i,
            quantity_f,
            quantity_b,
            position,
            density,
            h,
            simulation_params,
        );

        sum + boundary_divergence
    }

    fn calculate_pressure_accel_divergence_from_pressures_inefficiently(
        i: usize,
        density: &[FT],
        position: &[VF<D>],
        mass: &[FT],
        pressure: impl Fn(usize) -> FT + Copy,
        h: &[FT],
        neighs: &NeighborhoodCache,
        boundary_handler: &BoundaryHandler<DU, D>,
        simulation_params: SimulationParams,
    ) -> FT {
        let calculate_pressure_accel_from_pressure_fn = |i| {
            Self::calculate_particle_pressure_accel(
                i,
                position,
                mass,
                pressure,
                density,
                h,
                neighs,
                boundary_handler,
                simulation_params,
            )
        };

        Self::calculate_divergence_iisph(
            i,
            calculate_pressure_accel_from_pressure_fn,
            zero(),
            position,
            mass,
            density,
            h,
            neighs,
            boundary_handler,
            simulation_params,
        )
    }

    fn calculate_source_term_divergence(
        i: usize,
        density: &[FT],
        velocity: &[VF<D>],
        position: &[VF<D>],
        mass: &[FT],
        h: &[FT],
        neighs: &NeighborhoodCache,
        boundary_handler: &BoundaryHandler<DU, D>,
        simulation_params: SimulationParams,
        dt: FT,
    ) -> FT {
        let velocity_adv_div = Self::calculate_divergence_iisph(
            i,
            |j| velocity[j],
            VF::zeros(),
            position,
            mass,
            density,
            h,
            neighs,
            boundary_handler,
            simulation_params,
        );

        -velocity_adv_div / dt
    }

    fn calculate_source_term_only_density_part(
        i: usize,
        density: &[FT],
        simulation_params: SimulationParams,
        dt: FT,
    ) -> FT {
        // let next_density_estimate = density[i];
        let next_density_estimate = match simulation_params.operator_discretization {
            OperatorDiscretization::ConsistentSimpleGradient | OperatorDiscretization::ConsistentSymmetricGradient => {
                density[i]
            }
            OperatorDiscretization::Winchenbach2020 => simulation_params.rest_density,
        };

        -(simulation_params.rest_density - density[i]) / (next_density_estimate * dt * dt)
    }

    fn calculate_source_term_full_with_omega(
        i: usize,
        density: &[FT],
        velocity: &[VF<D>],
        position: &[VF<D>],
        mass: &[FT],
        h: &[FT],
        neighs: &NeighborhoodCache,
        omega: &[FT],
        boundary_handler: &BoundaryHandler<DU, D>,
        simulation_params: SimulationParams,
        dt: FT,
    ) -> FT {
        let velocity_adv_div = Self::calculate_divergence_iisph(
            i,
            |j| velocity[j],
            VF::zeros(),
            position,
            mass,
            density,
            h,
            neighs,
            boundary_handler,
            simulation_params,
        );

        // let next_density_estimate = density[i];
        let next_density_estimate = simulation_params.rest_density;

        let source_term = -(simulation_params.rest_density - density[i]) / (next_density_estimate * dt * dt)
            - velocity_adv_div / (dt * omega[i]);
        source_term
    }

    fn calculate_source_term_full(
        i: usize,
        density: &[FT],
        velocity: &[VF<D>],
        position: &[VF<D>],
        mass: &[FT],
        h: &[FT],
        neighs: &NeighborhoodCache,
        boundary_handler: &BoundaryHandler<DU, D>,
        simulation_params: SimulationParams,
        dt: FT,
    ) -> FT {
        let velocity_adv_div = Self::calculate_divergence_iisph(
            i,
            |j| velocity[j],
            VF::zeros(),
            position,
            mass,
            density,
            h,
            neighs,
            boundary_handler,
            simulation_params,
        );

        // let next_density_estimate = density[i];
        let next_density_estimate = match simulation_params.operator_discretization {
            OperatorDiscretization::ConsistentSimpleGradient | OperatorDiscretization::ConsistentSymmetricGradient => {
                density[i]
            }
            OperatorDiscretization::Winchenbach2020 => simulation_params.rest_density,
        };

        let source_term =
            -(simulation_params.rest_density - density[i]) / (next_density_estimate * dt * dt) - velocity_adv_div / dt;
        source_term
    }

    #[inline]
    fn calculate_particle_pressure_accel(
        i: usize,
        position: &[VF<D>],
        mass: &[FT],
        pressure: impl Fn(usize) -> FT + Copy,
        density: &[FT],
        h: &[FT],
        neighs: &NeighborhoodCache,
        boundary_handler: &BoundaryHandler<DU, D>,
        simulation_params: SimulationParams,
    ) -> VF<D> {
        let fluid_accel = Self::calculate_fluid_fluid_pressure_accel(
            i,
            position,
            mass,
            pressure,
            density,
            h,
            neighs,
            simulation_params,
        );

        let boundary_accel =
            boundary_handler.iisph_boundary_pressure_accel(i, position, mass, pressure, density, h, simulation_params);

        fluid_accel + boundary_accel
    }

    #[inline]
    fn calculate_fluid_fluid_pressure_accel(
        i: usize,
        position: &[VF<D>],
        mass: &[FT],
        pressure: impl Fn(usize) -> FT,
        density: &[FT],
        h: &[FT],
        neighs: &NeighborhoodCache,
        simulation_params: SimulationParams,
    ) -> VF<D> {
        assert!(density[i] > 0.0000001);

        let p1_pressure_term = pressure(i) / (density[i] * density[i]);

        // compute common SPH pressure acceleration (e.g. [IISPH Eq 2 divided by m_i] and [WCSPH Eq 6])
        let mut pressure_accel_acc: VF<D> = zero();
        for j in neighs.iter(i) {
            assert!(density[j] > 0.0000001);

            let h_ij = smoothing_length(h, i, j, simulation_params);
            let dg_ij: VF<D> = DU::kernel_derivh(position[i] - position[j], h_ij);

            let p2_pressure_term = pressure(j) / (density[j] * density[j]);

            pressure_accel_acc += -mass[j] * (p1_pressure_term + p2_pressure_term) * dg_ij;
        }
        assert_vector_non_nan(&pressure_accel_acc, "pressure_accel_acc");
        pressure_accel_acc
    }

    fn check_correct_neighborhood(
        i: usize,
        position: &[VF<D>],
        h: &[FT],
        neighs: &NeighborhoodCache,
        simulation_params: SimulationParams,
    ) {
        let xi = position[i];
        let mut neighs_correct: HashSet<usize> = HashSet::new();
        for (j, xj) in position.iter().enumerate() {
            let sr_ij = support_radius::<DU, D>(h, i, j, simulation_params);
            if (xi - xj).norm_squared() < sr_ij * sr_ij {
                neighs_correct.insert(j);
            }
        }
        let mut neighs_found: HashSet<usize> = HashSet::new();
        for j in neighs.iter(i) {
            neighs_found.insert(j);
        }

        for &found_but_incorrect in neighs_found.difference(&neighs_correct) {
            let xj = position[found_but_incorrect];
            let sr_ij = support_radius::<DU, D>(h, i, found_but_incorrect, simulation_params);
            let dist_ij_sq = (xi - xj).norm_squared();
            let sr_ij_sq = sr_ij * sr_ij;
            if dist_ij_sq != sr_ij_sq {
                panic!(
                    "found neighbor j={} for particle i={} but it is incorrect | x_ij^2={} | sr_ij^2={} | equal={}",
                    found_but_incorrect,
                    i,
                    dist_ij_sq,
                    sr_ij_sq,
                    dist_ij_sq == sr_ij_sq
                );
            }
        }

        for &correct_but_not_found in neighs_correct.difference(&neighs_found) {
            let xj = position[correct_but_not_found];
            let sr_ij = support_radius::<DU, D>(h, i, correct_but_not_found, simulation_params);
            let dist_ij_sq = (xi - xj).norm_squared();
            let sr_ij_sq = sr_ij * sr_ij;
            if dist_ij_sq != sr_ij_sq {
                panic!(
                    "did not find correct neighbor j={} for particle i={} | x_ij^2={} | sr_ij^2={} | equal={}",
                    correct_but_not_found,
                    i,
                    dist_ij_sq,
                    sr_ij_sq,
                    dist_ij_sq == sr_ij_sq
                );
            }
        }
    }

    fn h_next_from_mass(h: &mut Vec<FT>, mass: &[FT], simulation_params: SimulationParams) {
        par_iter_mut1(h, |i, p_h| {
            *p_h = local_smoothing_length_from_mass::<DU, D>(mass[i], simulation_params.rest_density);

            // "Constrained Neighbor Lists for SPH-based Fluid Simulations Eq." 3
        });
    }

    fn estimate_h_next_from_distribution(
        h_next: &mut Vec<FT>,
        h: &Vec<FT>,
        mass: &[FT],
        position: &[VF<D>],
        boundary_handler: &BoundaryHandler<DU, D>,
        neighs: &NeighborhoodCache,
        simulation_params: SimulationParams,
        clamping_factor_opt: Option<FT>,
    ) {
        par_iter_mut1(h_next, |i, p_h_next| {
            let w = 0.5;
            let volume_estimate;
            let mut w_sum: FT = 0.;
            for j in neighs.iter(i) {
                let h_ij = smoothing_length(h, i, j, simulation_params);
                w_sum += DU::kernelh(position[i] - position[j], h_ij);
            }

            let boundary_volume = match boundary_handler {
                BoundaryHandler::BoundaryWinchenbach2020(b) => b.lambda_sum(i),
                BoundaryHandler::ParticleBasedBoundaryHandler(_b) => {
                    todo!()
                }
                BoundaryHandler::NoBoundaryHandler(_b) => 0.,
            };

            // "Constrained Neighbor Lists for SPH-based Fluid Simulations" Eq. 4
            // let vi = mass[i] / simulation_params.rest_density;
            // volume_estimate = vi / (vi * w_sum + boundary_volume);
            volume_estimate = (1. - FT::min(boundary_volume, 0.5)) / w_sum;
            // volume_estimate = 1. / w_sum;
            if volume_estimate < 0. {
                panic!(
                    "volume_estimate:{}<0. boundary_volume:{} w_sum:{}",
                    volume_estimate, boundary_volume, w_sum
                );
            }
            assert!(volume_estimate >= 0.);

            let h_new = ETA * DU::sphere_volume_to_radius(volume_estimate);
            let h_old = h[i];

            // let h_max = local_smoothing_length_from_mass::<DU, D>(mass[i], simulation_params.rest_density) * 1.;

            *p_h_next = w * h_new + (1. - w) * h_old;

            if let Some(clamping_factor) = clamping_factor_opt {
                *p_h_next = FT::min(
                    *p_h_next,
                    clamping_factor
                        * local_smoothing_length_from_mass::<DU, D>(mass[i], simulation_params.rest_density),
                );
            }

            // "Constrained Neighbor Lists for SPH-based Fluid Simulations Eq." 3
        });
    }

    fn estimate_h_next_from_distribution2(
        h_next: &mut Vec<FT>,
        h: &Vec<FT>,
        mass: &[FT],
        position: &[VF<D>],
        boundary_handler: &BoundaryHandler<DU, D>,
        neighs: &NeighborhoodCache,
        simulation_params: SimulationParams,
    ) {
        par_iter_mut1(h_next, |i, p_h_next| {
            let w = 0.5;
            let volume_estimate;
            let mut v_w_sum: FT = 0.;
            for j in neighs.iter(i) {
                let h_ij = smoothing_length(h, i, j, simulation_params);
                let vj = mass[j] / simulation_params.rest_density;
                v_w_sum += vj * DU::kernelh(position[i] - position[j], h_ij);
            }

            let boundary_volume = match boundary_handler {
                BoundaryHandler::BoundaryWinchenbach2020(b) => b.lambda_sum(i),
                BoundaryHandler::ParticleBasedBoundaryHandler(_b) => {
                    todo!()
                }
                BoundaryHandler::NoBoundaryHandler(_b) => 0.,
            };

            let vi = mass[i] / simulation_params.rest_density;

            // "Constrained Neighbor Lists for SPH-based Fluid Simulations" Eq. 4
            volume_estimate = vi / (v_w_sum + boundary_volume);
            assert!(volume_estimate >= 0.);

            let h_new = ETA * DU::sphere_volume_to_radius(volume_estimate);
            let h_old = h[i];

            *p_h_next = w * h_new + (1. - w) * h_old;
            // "Constrained Neighbor Lists for SPH-based Fluid Simulations Eq." 3
        });
    }

    pub fn single_step(&mut self, simulation_params: SimulationParams) {
        let dt = self.single_step_without_adaptivity(simulation_params);
        if PARTICLE_SIZES == ParticleSizes::Adaptive {
            self.single_step_adaptivity(simulation_params, dt);
        }
    }

    pub fn single_step_without_adaptivity(&mut self, simulation_params: SimulationParams) -> FT {
        println!("");
        println!("==================================================");
        println!(
            "================= Begin Step {} Time {} ==================",
            self.step_number, self.time
        );
        println!("==================================================");
        println!("");

        self.vcounters
            .add_value("particle-count", self.particles.position.len() as FT);

        self.pcounters.begin("simulation-step");

        let particles = &mut self.particles;
        let num_fluid_particles = particles.position.len();

        // update kernel support radius
        if PARTICLE_SIZES == ParticleSizes::Adaptive {
            match simulation_params.support_length_estimation {
                SupportLengthEstimation::FromMass => {
                    Self::h_next_from_mass(&mut particles.h2, &particles.mass, simulation_params);
                }
                SupportLengthEstimation::FromDistribution
                | SupportLengthEstimation::FromDistributionClamped1
                | SupportLengthEstimation::FromDistributionClamped2
                | SupportLengthEstimation::FromDistribution2 => {
                    // the estimation from distribution needs a valid neighborhood first
                    //  -> "self.neighs" is invalidated after position integration and
                    //     even more so through particle splitting/merging
                    //
                    // only apply the support length that was estimated in the last step
                    mem::swap(&mut particles.h2, &mut particles.h2_next);
                }
            }
        }

        if !simulation_params.level_estimation_after_advection {
            // find fluid neighbors for fluid particles
            assert!(simulation_params.use_extended_range_for_level_estimation);
            assert!(simulation_params.level_estimation_method != LevelEstimationMethod::CenterDiff, "center diff level estimation method needs density values which are not available when performing level estimation as first step in loop");

            self.pcounters.begin("neighborhood");
            build_neighborhood_list::<DU, D>(
                simulation_params,
                &particles.position,
                &particles.h2,
                simulation_params.level_estimation_range / ETA,
                &mut self.neighs,
            );
            self.pcounters.end("neighborhood");

            self.pcounters.begin("level-estimation");
            Self::perform_level_estimation(
                simulation_params,
                &particles.mass,
                &particles.position,
                &particles.density,
                &particles.h2,
                // dt,
                &mut particles.stash,
                &mut particles.level_estimation,
                &mut particles.level_estimation_temp,
                &self.neighs,
                &self.boundary_handler,
                &mut particles.flag_is_fluid_surface,
                &mut particles.flag_insufficient_neighs,
            );
            self.pcounters.end("level-estimation");

            self.pcounters.begin("neighborhood");
            self.neighs.filter_down::<DU, D>(
                simulation_params,
                &particles.h2,
                &particles.position,
                DU::support_radius_by_smoothing_length(),
            );
            self.pcounters.end_add_to_last("neighborhood");
        } else {
            // find fluid neighbors for fluid particles
            self.pcounters.begin("neighborhood");
            build_neighborhood_list::<DU, D>(
                simulation_params,
                &particles.position,
                &particles.h2,
                DU::support_radius_by_smoothing_length(),
                &mut self.neighs,
            );
            self.pcounters.end("neighborhood");
        }

        par_iter_mut1(&mut particles.neighbor_count, |i, p_neighbor_count| {
            *p_neighbor_count = self.neighs.neighbor_count(i);
        });

        if simulation_params.check_neighborhood {
            println!("=====> SLOW: CHECK NEIGHBORHOOD OF ALL PARTICLES <=====");

            for i in 0..num_fluid_particles {
                Self::check_correct_neighborhood(
                    i,
                    &particles.position,
                    &particles.h2,
                    &self.neighs,
                    simulation_params,
                );
            }
        }

        if PARTICLE_SIZES == ParticleSizes::Adaptive {
            match simulation_params.support_length_estimation {
                SupportLengthEstimation::FromMass => {
                    // nothing to do here
                }
                SupportLengthEstimation::FromDistribution => {
                    Self::estimate_h_next_from_distribution(
                        &mut particles.h2_next,
                        &particles.h2,
                        &particles.mass,
                        &particles.position,
                        &self.boundary_handler,
                        &self.neighs,
                        simulation_params,
                        None,
                    );
                }
                SupportLengthEstimation::FromDistributionClamped1 => {
                    Self::estimate_h_next_from_distribution(
                        &mut particles.h2_next,
                        &particles.h2,
                        &particles.mass,
                        &particles.position,
                        &self.boundary_handler,
                        &self.neighs,
                        simulation_params,
                        Some(1.),
                    );
                }
                SupportLengthEstimation::FromDistributionClamped2 => {
                    Self::estimate_h_next_from_distribution(
                        &mut particles.h2_next,
                        &particles.h2,
                        &particles.mass,
                        &particles.position,
                        &self.boundary_handler,
                        &self.neighs,
                        simulation_params,
                        Some(2.),
                    );
                }
                SupportLengthEstimation::FromDistribution2 => {
                    Self::estimate_h_next_from_distribution2(
                        &mut particles.h2_next,
                        &particles.h2,
                        &particles.mass,
                        &particles.position,
                        &self.boundary_handler,
                        &self.neighs,
                        simulation_params,
                    );
                }
            }
        }

        if PARTICLE_SIZES == ParticleSizes::Adaptive && simulation_params.constrain_neighborhood_count {
            // reduce number of neighbors
            let target_neighbors = optimal_neighbor_number::<DU, D>() as usize + 5;
            par_iter_mut2(
                &mut particles.h2_next,
                &mut particles.flag_neighborhood_reduced,
                |i, p_h_next, p_flag_neighborhood_reduced| {
                    if self.neighs.neighbor_count(i) > target_neighbors {
                        let mut fringe_list: Vec<FT> = Vec::new();
                        for j in self.neighs.iter(i) {
                            let xi = particles.position[i];
                            let xj = particles.position[j];
                            let x_ij = (xi - xj).norm();
                            let srj = support_radius_single::<DU, D>(&particles.h2, j, simulation_params);
                            fringe_list.push(2. * x_ij - srj);
                        }
                        fringe_list.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap());
                        *p_h_next = fringe_list[self.neighs.neighbor_count(i) - target_neighbors];
                        assert!(*p_h_next < smoothing_length_single(&particles.h2, i, simulation_params));
                        *p_flag_neighborhood_reduced = true;
                        assert!(*p_h_next >= 0.);
                    } else {
                        *p_h_next = particles.h2[i];
                        *p_flag_neighborhood_reduced = false;
                    }
                },
            );

            mem::swap(&mut particles.h2, &mut particles.h2_next);

            // TODO: remove all unnecessary correspondences that now have a support radius "(h[i] + h[j]) * 0.5"
            // which is smaller than the distance to each other
        }

        self.boundary_handler
            .update_after_advect(&particles.position, &particles.h2, simulation_params);

        let max_velocity_cfl_sq: FT = into_par_iter(0..num_fluid_particles)
            .map(|i| {
                let v = particles.velocity[i];
                let support_radius = support_radius_single::<DU, D>(&particles.h2, i, simulation_params);
                support_radius * support_radius / (v.norm_squared() + 0.01)
            })
            .min_by(|x, y| x.partial_cmp(y).unwrap())
            .unwrap();
        let cfl_dt = simulation_params.cfl_factor * max_velocity_cfl_sq.sqrt();
        let dt = FT::min(simulation_params.max_dt, cfl_dt);
        // let dt = simulation_params.dt_default;

        println!(
            "dt: {}ms (max dt: {}ms; CFL dt: {}ms (factor {})",
            dt * 1000.,
            simulation_params.max_dt * 1000.,
            cfl_dt * 1000.,
            simulation_params.cfl_factor
        );

        self.vcounters.add_value("dt", dt);

        Self::calculate_all_particle_densities(particles, simulation_params, &self.boundary_handler, &self.neighs);

        // let min_density = particles.density.iter().cloned().fold(FT::INFINITY, |a, b| a.min(b));
        // let max_density = particles.density.iter().cloned().fold(-FT::INFINITY, |a, b| a.max(b));
        // println!("MAX DENSITY: {} MIN DENSITY: {}", max_density, min_density);

        // count number of neighbors
        if false {
            let mut total_neigh_number = 0;
            let mut total_particle_number = 0;
            for neigh_list in self.neighs.internal_lists().iter() {
                // if particles.density[i] > 0.998 {
                total_neigh_number += neigh_list.len();
                total_particle_number += 1;
                // }
            }
            if total_particle_number > 0 {
                println!(
                    "Avg neighbor count: {} for {} particles",
                    total_neigh_number / total_particle_number,
                    total_particle_number
                );
            } else {
                println!("Avg neighbor count: not enough particles with rest density");
            }
        }

        if PARTICLE_SIZES == ParticleSizes::Uniform {
            println!("h:{}", simulation_params.h);
        }

        par_iter_mut1(&mut particles.constant_field, |i, p_constant_field| {
            *p_constant_field = 0.;
            for j in self.neighs.iter(i) {
                let x_ij = particles.position[i] - particles.position[j];
                let h_ij = smoothing_length(&particles.h2, i, j, simulation_params);
                let w_ij = DU::kernelh(x_ij, h_ij);
                *p_constant_field += particles.mass[j] / particles.density[j] * w_ij;
            }

            *p_constant_field +=
                self.boundary_handler
                    .density_boundary_term(i, &particles.position, &particles.h2, simulation_params)
                    / simulation_params.rest_density;
        });

        Self::compute_aii(
            &mut particles.aii,
            &particles.density,
            &particles.position,
            &particles.mass,
            &particles.h2,
            &self.neighs,
            &self.boundary_handler,
            simulation_params,
        );

        match simulation_params.pressure_solver_method {
            PressureSolverMethod::IISPH2 => {
                for i in 0..num_fluid_particles {
                    let mut omega: FT = 1.;

                    #[allow(non_snake_case)]
                    fn dwdh<const D: usize>(d: FT, H: FT) -> FT {
                        assert!(D == 2);
                        let q = d / H;

                        let cd = 40. / (7. * PI);
                        let w = cubic_kernel_unnormalized(q);
                        let wd = cubic_kernel_unnormalized_deriv(q);

                        cd * -(D as FT) / (H * H * H) * w + cd / (H * H) * wd * (-d / (H * H))
                    }

                    if particles.particle_size_class[i] == ParticleSizeClass::Large {
                        let h_ii = smoothing_length_single(&particles.h2, i, simulation_params);

                        #[allow(non_snake_case)]
                        let H_i = particles.h2[i] * DU::support_radius_by_smoothing_length();
                        #[allow(non_snake_case)]
                        let H_ii = h_ii * DU::support_radius_by_smoothing_length();

                        let d = 0.;

                        omega += H_i / (3. * particles.density[i]) * particles.mass[i] * dwdh::<D>(d, H_ii);
                    } else {
                        for j in self.neighs.iter(i) {
                            // let weight = DU::kernelh(position[particle_id] - position[neigh_particle_id], h_ij);
                            let x_ij = particles.position[i] - particles.position[j];
                            let h_ij = smoothing_length(&particles.h2, i, j, simulation_params);

                            #[allow(non_snake_case)]
                            let H_i = particles.h2[i] * DU::support_radius_by_smoothing_length();
                            #[allow(non_snake_case)]
                            let H_ij = h_ij * DU::support_radius_by_smoothing_length();

                            let d = x_ij.norm();

                            assert!(D == 2);

                            omega += H_i / (3. * particles.density[i]) * particles.mass[j] * dwdh::<D>(d, H_ij);
                        }
                    }

                    omega = FT::min(2.5, FT::max(omega, 0.125));

                    particles.omega[i] = omega;
                }

                Self::update_velocity_with_non_pressure_accel(
                    &mut particles.velocity,
                    &mut particles.velocity_temp,
                    &particles.density,
                    &particles.position,
                    &particles.mass,
                    &particles.h2,
                    &self.neighs,
                    simulation_params,
                    dt,
                );

                // pre-calculate IISPH parameters for pressure solve iterations
                par_iter_mut2(
                    &mut particles.pressure,
                    &mut particles.ppe_source_term,
                    |i, p_pressure, p_ppe_source_term| {
                        *p_pressure = 0.;
                        *p_ppe_source_term = Self::calculate_source_term_full_with_omega(
                            i,
                            &particles.density,
                            &particles.velocity,
                            &particles.position,
                            &particles.mass,
                            &particles.h2,
                            &self.neighs,
                            &particles.omega,
                            &self.boundary_handler,
                            simulation_params,
                            dt,
                        );
                    },
                );

                Self::iisph_pressure_iterations(
                    particles,
                    &self.neighs,
                    &self.boundary_handler,
                    simulation_params.iisph_max_avg_density_error,
                    PressureSolverResidualType::DensityError,
                    true,
                    simulation_params,
                    dt,
                );

                for i in 0..num_fluid_particles {
                    particles.pressure[i] /= particles.omega[i].sqrt();
                }

                Self::calculate_particle_pressure_accels(
                    &mut particles.pressure_accel,
                    &particles.position,
                    &particles.mass,
                    &particles.pressure,
                    &particles.density,
                    &particles.h2,
                    &self.neighs,
                    &self.boundary_handler,
                    simulation_params,
                );

                par_iter_mut2(
                    &mut particles.position,
                    &mut particles.velocity,
                    |i, p_position, p_velocity| {
                        // *p_position += dt * dt * particles.pressure_accel[i];

                        *p_velocity += dt * particles.pressure_accel[i];
                        *p_position += dt * (*p_velocity);
                        // *p_position += dt * (*p_velocity) + dt * dt * particles.pressure_accel[i];

                        assert_vector_non_nan(&p_velocity, "p_velocity");
                    },
                );
            }

            PressureSolverMethod::IISPH => {
                Self::update_velocity_with_non_pressure_accel(
                    &mut particles.velocity,
                    &mut particles.velocity_temp,
                    &particles.density,
                    &particles.position,
                    &particles.mass,
                    &particles.h2,
                    &self.neighs,
                    simulation_params,
                    dt,
                );

                Self::prepare_full_ppe(
                    &mut particles.pressure,
                    &mut particles.ppe_source_term,
                    &particles.density,
                    &particles.position,
                    &particles.velocity,
                    &particles.mass,
                    &particles.h2,
                    &self.neighs,
                    &self.boundary_handler,
                    simulation_params,
                    dt,
                );

                Self::iisph_pressure_iterations(
                    particles,
                    &self.neighs,
                    &self.boundary_handler,
                    simulation_params.iisph_max_avg_density_error,
                    PressureSolverResidualType::DensityError,
                    true,
                    simulation_params,
                    dt,
                );

                for i in 0..particles.position.len() {
                    if particles.aii[i] < 0. {
                        panic!("ABORT ADVECT DUE TO NEGATIVE M_II");
                    }
                }

                par_iter_mut2(
                    &mut particles.position,
                    &mut particles.velocity,
                    |i, p_position, p_velocity| {
                        // *p_position += dt * dt * particles.pressure_accel[i];

                        *p_velocity += dt * particles.pressure_accel[i];
                        *p_position += dt * (*p_velocity);
                        // *p_position += dt * (*p_velocity) + dt * dt * particles.pressure_accel[i];

                        assert_vector_non_nan(&p_velocity, "p_velocity");
                    },
                );
            }

            PressureSolverMethod::OnlyDivergence => {
                Self::update_velocity_with_non_pressure_accel(
                    &mut particles.velocity,
                    &mut particles.velocity_temp,
                    &particles.density,
                    &particles.position,
                    &particles.mass,
                    &particles.h2,
                    &self.neighs,
                    simulation_params,
                    dt,
                );

                Self::prepare_ppe_divergence(
                    &mut particles.pressure,
                    &mut particles.ppe_source_term,
                    &particles.density,
                    &particles.position,
                    &particles.velocity,
                    &particles.mass,
                    &particles.h2,
                    &self.neighs,
                    &self.boundary_handler,
                    simulation_params,
                    dt,
                );

                Self::iisph_pressure_iterations(
                    particles,
                    &self.neighs,
                    &self.boundary_handler,
                    simulation_params.hybrid_dfsph_max_avg_divergence_error,
                    PressureSolverResidualType::DivergenceError,
                    true,
                    simulation_params,
                    dt,
                );

                par_iter_mut2(
                    &mut particles.position,
                    &mut particles.velocity,
                    |i, p_position, p_velocity| {
                        // *p_position += dt * dt * particles.pressure_accel[i];

                        // *p_position += dt * (*p_velocity) + dt * dt * particles.pressure_accel[i];

                        *p_velocity += dt * particles.pressure_accel[i];
                        *p_position += dt * (*p_velocity);

                        assert_vector_non_nan(&p_velocity, "p_velocity");
                    },
                );
            }

            PressureSolverMethod::HybridDFSPH => {
                if simulation_params.hybrid_dfsph_non_pressure_accel_before_divergence_free {
                    Self::update_velocity_with_non_pressure_accel(
                        &mut particles.velocity,
                        &mut particles.velocity_temp,
                        &particles.density,
                        &particles.position,
                        &particles.mass,
                        &particles.h2,
                        &self.neighs,
                        simulation_params,
                        dt,
                    );
                }

                self.pcounters.begin("div-solver");
                Self::prepare_ppe_divergence(
                    &mut particles.pressure,
                    &mut particles.ppe_source_term,
                    &particles.density,
                    &particles.position,
                    &particles.velocity,
                    &particles.mass,
                    &particles.h2,
                    &self.neighs,
                    &self.boundary_handler,
                    simulation_params,
                    dt,
                );

                let (div_iters, _) = Self::iisph_pressure_iterations(
                    particles,
                    &self.neighs,
                    &self.boundary_handler,
                    simulation_params.hybrid_dfsph_max_avg_divergence_error,
                    PressureSolverResidualType::DivergenceError,
                    true,
                    simulation_params,
                    dt,
                );
                if div_iters > 0 {
                    self.vcounters.add_value("div-iterations", div_iters as FT);
                }
                self.pcounters.end("div-solver");

                par_iter_mut2(
                    &mut particles.position,
                    &mut particles.velocity,
                    |i, _p_position, p_velocity| {
                        // *p_position += dt * dt * particles.pressure_accel[i];

                        // *p_position += dt * (*p_velocity) + dt * dt * particles.pressure_accel[i];

                        *p_velocity += dt * particles.pressure_accel[i];
                        // *p_position += dt * (*p_velocity);

                        assert_vector_non_nan(&p_velocity, "p_velocity");
                    },
                );

                if !simulation_params.hybrid_dfsph_non_pressure_accel_before_divergence_free {
                    Self::update_velocity_with_non_pressure_accel(
                        &mut particles.velocity,
                        &mut particles.velocity_temp,
                        &particles.density,
                        &particles.position,
                        &particles.mass,
                        &particles.h2,
                        &self.neighs,
                        simulation_params,
                        dt,
                    );
                }

                // if self.step_number == 1000 { panic!("stop"); }

                self.pcounters.begin("density-solver");
                match simulation_params.hybrid_dfsph_density_source_term {
                    HybridDfsphDensitySourceTerm::DensityAndDivergence => {
                        Self::prepare_full_ppe(
                            &mut particles.pressure,
                            &mut particles.ppe_source_term,
                            &particles.density,
                            &particles.position,
                            &particles.velocity,
                            &particles.mass,
                            &particles.h2,
                            &self.neighs,
                            &self.boundary_handler,
                            simulation_params,
                            dt,
                        );
                    }

                    HybridDfsphDensitySourceTerm::OnlyDensity => {
                        Self::prepare_only_density_part_ppe(
                            &mut particles.pressure,
                            &mut particles.ppe_source_term,
                            &particles.density,
                            simulation_params,
                            dt,
                        );
                    }
                }

                let (density_iters, _) = Self::iisph_pressure_iterations(
                    particles,
                    &self.neighs,
                    &self.boundary_handler,
                    simulation_params.hybrid_dfsph_max_avg_density_error,
                    PressureSolverResidualType::DensityError,
                    true,
                    simulation_params,
                    dt,
                );
                if density_iters > 0 {
                    self.vcounters.add_value("density-iterations", density_iters as FT);
                }
                self.pcounters.end("density-solver");

                par_iter_mut2(
                    &mut particles.position,
                    &mut particles.velocity,
                    |i, p_position, p_velocity| {
                        // *p_position += dt * dt * particles.pressure_accel[i];

                        // let h = match PARTICLE_SIZES {
                        //     ParticleSizes::Adaptive => {
                        //         particles.h[i]
                        //     }
                        //     ParticleSizes::Uniform => {
                        //         simulation_params.kernel_support_radius
                        //     }
                        // };
                        // let mut dx = dtfake * dtfake * particles.pressure_accel[i];
                        // let dxn = dx.norm();
                        // let allowed_displacement = h * 0.1;
                        // if dxn > allowed_displacement {
                        //     dx = dx * allowed_displacement / dxn;
                        // }
                        // *p_position += dt * (*p_velocity) + dx;

                        *p_position += dt * (*p_velocity) + dt * dt * particles.pressure_accel[i];
                        *p_velocity +=
                            dt * particles.pressure_accel[i] * FT::min(dt * simulation_params.hybrid_dfsph_factor, 1.);

                        /*
                         * FINE for small timesteps because divergence free solver is enough
                         * LARGE timesteps -> oscillations at the boundary because divergence
                         * free solver would move particles into boundary and with only position
                         * updates they are "teleported" back (BAD)
                         */
                        // *p_velocity += dt * particles.pressure_accel[i];

                        // |a_p| is in O(1/dt^2)
                        // v_p = dt|a_p| is in O(1/dt)
                        // v_p = min(c * dt * dt|a_p|, 1) is in O(1/dt)

                        /*
                         * FINE for large time steps
                         */
                        // *p_velocity += dt * particles.pressure_accel[i];

                        // *p_position += dt * (*p_velocity);

                        assert_vector_non_nan(p_position, "p_position");
                    },
                );
            }
        }

        if simulation_params.viscosity_type == ViscosityType::XSPH {
            // perform XSPH velocity smoothing
            todo!()
        }

        if simulation_params.level_estimation_after_advection {
            // find fluid neighbors for fluid particles
            if simulation_params.use_extended_range_for_level_estimation {
                build_neighborhood_list::<DU, D>(
                    simulation_params,
                    &particles.position,
                    &particles.h2,
                    simulation_params.level_estimation_range / ETA,
                    &mut self.neighs,
                );
            }

            self.pcounters.begin("level-estimation");
            Self::perform_level_estimation(
                simulation_params,
                &particles.mass,
                &particles.position,
                &particles.density,
                &particles.h2,
                // dt,
                &mut particles.stash,
                &mut particles.level_estimation,
                &mut particles.level_estimation_temp,
                &self.neighs,
                &self.boundary_handler,
                &mut particles.flag_is_fluid_surface,
                &mut particles.flag_insufficient_neighs,
            );
            self.pcounters.end("level-estimation");
        }

        self.pcounters.begin("level-estimation");
        Self::smooth_level_estimation_field(
            simulation_params,
            &particles.mass,
            &particles.position,
            &particles.density,
            &particles.h2,
            // dt,
            &mut particles.level_estimation,
            &mut particles.level_estimation_temp,
            &mut particles.level_old,
            &self.neighs,
        );
        self.pcounters.end_add_to_last("level-estimation");

        self.time += dt;
        self.step_number += 1;

        self.pcounters.end("simulation-step");

        dt
    }

    pub fn single_step_adaptivity(&mut self, simulation_params: SimulationParams, dt: FT) {
        self.pcounters.begin("simulation-step");
        self.pcounters.begin("adaptivity");

        let particles = &mut self.particles;

        // build_neighborhood_list::<DU, D>
        //     simulation_params,
        //     &particles.position,
        //     &particles.h,
        //     &mut self.neighs,
        // );

        let total_mass1: FT = particles.mass.iter().cloned().sum();

        if simulation_params.sharing {
            println!("share");
            classify_particles::<DU, D>(particles, &self.neighs, simulation_params);
            find_share_partner_sequential::<DU, D>(particles, &self.neighs, simulation_params, dt);
            share_particles::<DU, D>(
                particles,
                &mut self.neighs,
                &mut self.boundary_handler,
                simulation_params,
                dt,
            );
        }

        match self.step_number % 2 {
            0 => {
                if simulation_params.merging {
                    println!("merge");
                    classify_particles::<DU, D>(particles, &self.neighs, simulation_params);
                    find_merge_partner_sequential::<DU, D>(particles, &self.neighs, dt, simulation_params);
                    merge_particles::<DU, D>(
                        particles,
                        &mut self.neighs,
                        &mut self.boundary_handler,
                        dt,
                        simulation_params,
                    );
                }
            }
            _ => {
                if simulation_params.splitting {
                    println!("split");
                    classify_particles::<DU, D>(particles, &self.neighs, simulation_params);
                    split_particles::<DU, D>(
                        particles,
                        &mut self.neighs,
                        &mut self.boundary_handler,
                        simulation_params,
                        &self.split_patterns,
                        dt,
                    );
                }
            }
        }

        let total_mass2: FT = particles.mass.iter().cloned().sum();
        assert_ft_approx_eq(total_mass1, total_mass2, 0.005, || format!("mass sum"));

        self.pcounters.end("adaptivity");
        self.pcounters.end_add_to_last("simulation-step");
    }
}

use crate::boundary_handler::NoBoundaryHandler;

pub static INIT_VISUALIZED_ATTRIBUTE: VisualizedAttribute = VisualizedAttribute::Velocity;

#[derive(PartialEq, Eq, Debug, Clone, Copy, Serialize, Deserialize)]
pub enum VisualizedAttribute {
    Distance,
    SingleColor,
    ParticleSizeClass,
    Pressure,
    Density,
    Velocity,
    RandomColor,
    Aii,
    NeighborCount,
    MinDistanceToNeighbor,
    ConstantField,
    SourceTerm,
}

pub const ALL_VISUALIZED_ATTIBUTES: [VisualizedAttribute; 12] = [
    VisualizedAttribute::Distance,
    VisualizedAttribute::SingleColor,
    VisualizedAttribute::ParticleSizeClass,
    VisualizedAttribute::Pressure,
    VisualizedAttribute::Density,
    VisualizedAttribute::Velocity,
    VisualizedAttribute::RandomColor,
    VisualizedAttribute::Aii,
    VisualizedAttribute::NeighborCount,
    VisualizedAttribute::MinDistanceToNeighbor,
    VisualizedAttribute::ConstantField,
    VisualizedAttribute::SourceTerm,
];

impl VisualizedAttribute {
    pub fn as_str_lowercase(&self) -> &'static str {
        match self {
            &Self::Aii => "a_ii",
            &Self::Pressure => "pressure",
            &Self::Density => "density",
            &Self::Distance => "distance to surface",
            &Self::Velocity => "velocity",
            &Self::NeighborCount => "neighborhood count",
            &Self::RandomColor => "random color",
            &Self::MinDistanceToNeighbor => "distance to neighbor",
            &Self::ParticleSizeClass => "particle size class",
            &Self::SingleColor => "single color",
            &Self::ConstantField => "constant field",
            &Self::SourceTerm => "source term",
        }
    }
}

#[derive(PartialEq, Eq, Copy, Clone, Debug, Serialize, Deserialize)]
pub enum DrawShape {
    // fast
    Dot,

    // slower
    Circle,

    // slowest
    FilledCircle,
    FilledCircleWithBorder,
    FilledCircleWithAABorder,
    Cairo,

    Metaball,
}
impl DrawShape {
    fn filled_circle_with_border() -> DrawShape {
        DrawShape::FilledCircleWithBorder
    }
}

#[derive(Copy, Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationParams {
    pub visualized_attribute: VisualizedAttribute,
    #[serde(default = "DrawShape::filled_circle_with_border")]
    pub draw_shape: DrawShape,
    #[serde(default)]
    pub draw_support_radius: bool,
    #[serde(default)]
    pub show_flag_is_fluid_surface: bool,
    #[serde(default)]
    pub show_flag_neighborhood_reduced: bool,
    #[serde(default)]
    pub take_data_from_stash: bool,
}

impl Default for VisualizationParams {
    fn default() -> Self {
        Self {
            visualized_attribute: INIT_VISUALIZED_ATTRIBUTE,
            draw_shape: DrawShape::FilledCircle,
            draw_support_radius: false,
            show_flag_neighborhood_reduced: false,
            show_flag_is_fluid_surface: false,
            take_data_from_stash: false,
        }
    }
}

pub trait SimulationVisualizer<DU: DimensionUtils<D>, const D: usize> {
    fn present(
        &mut self,
        particles: &ParticleVec<D>,
        neighs: &NeighborhoodCache,
        boundary_handler: &BoundaryHandler<DU, D>,
        simulation_params: &mut SimulationParams,
        visualization_params: VisualizationParams,
        simulation_failed: bool,
    ) -> Result<bool, String>;
}

fn add_fluid_block<DU: DimensionUtils<D>, const D: usize>(
    min: VF<2>,
    max: VF<2>,
    grid_spacing: FT,
    volume_fill_ratio: FT,
    init_with_velocity: VF<2>,
    particle_positions: &mut Vec<VF<2>>,
    particle_masses: &mut Vec<FT>,
    particle_velocities: &mut Vec<VF<2>>,
) {
    #[allow(unused)]
    fn find_optimal_mass<DU: DimensionUtils<D>, const D: usize>(mut mass: FT, spacing: FT, rest_density: FT) -> FT {
        let mut mass_update = mass * 0.9;
        let mut num_iter = 0;
        for _i in 0..40 {
            // calculate density for grid setting with given mass
            let h = local_smoothing_length_from_mass::<DU, D>(mass, rest_density);
            let sr = h * DU::support_radius_by_smoothing_length();
            let ri = FT::ceil(sr / spacing) as i32;
            let mut density = 0.;
            DU::iterate_grid_neighbors(ri, |v| {
                density += mass * DU::kernelh(v.map(|x| x as FT) * spacing, h);
            });

            num_iter += 1;

            println!(
                "iter {}: density={} rest={} mass={}",
                num_iter, density, rest_density, mass
            );
            if (density - rest_density).abs() < 0.000001 {
                // panic!("Num mass find iter: {}", num_iter);
                return mass;
            }

            // optimize mass
            if density > rest_density {
                mass -= mass_update;
            } else {
                mass += mass_update;
            }
            mass_update *= 0.5;
        }

        panic!("too many iterations")
    }

    let particle_volume = grid_spacing * grid_spacing * volume_fill_ratio;
    let particle_mass = particle_volume * INIT_REST_DENSITY;

    // let particle_mass = find_optimal_mass::<DU, D>(particle_mass, grid_spacing, INIT_REST_DENSITY);
    // println!("{} {}", particle_mass, particle_mass2);

    let box_size = max - min;

    let num_particles_x = (box_size.x / grid_spacing).floor() as usize;
    let num_particles_y = (box_size.y / grid_spacing).floor() as usize;

    for x in 0..num_particles_x {
        for y in 0..num_particles_y {
            particle_positions.push(vec2f(
                x as FT * grid_spacing + min.x, // - y as FT * 0.0001,
                y as FT * grid_spacing + min.y,
            ));
            particle_masses.push(particle_mass);
            particle_velocities.push(init_with_velocity);
        }
    }
}

pub fn generate_split_patterns(max_num_children: usize) -> SplitPatterns<2> {
    assert!(max_num_children > 1);

    let split_patterns: Vec<SplitPattern<2>> = into_par_iter(2..max_num_children)
        .map(|i| precalculate_split_pattern::<DimensionUtils2d, 2>(i, 1., false))
        .collect();

    SplitPatterns::new(split_patterns)
}

pub fn write_split_patterns_to_file(path: &Path, split_patterns: &SplitPatterns<2>) {
    let split_pattern_yaml = serde_yaml::to_string(split_patterns).unwrap();
    std::fs::write(path, split_pattern_yaml).unwrap();
}

pub fn load_split_patterns_from_file(path: &Path) -> SplitPatterns<2> {
    let split_pattern_yaml = std::fs::read_to_string(path).unwrap();
    let split_patterns: SplitPatterns<2> = serde_yaml::from_str(&split_pattern_yaml).unwrap();
    split_patterns
}

/*pub fn write_split_patterns_to_dir(path: &Path, split_patterns: &SplitPatterns<2>) {
    remove_dir_all(path).ok();
    create_dir_all(path).unwrap();
    for i in 2..=split_patterns.get_max_num_children() {
        let num_children = i + 2;

        let split_pattern_yaml = serde_yaml::to_string(split_patterns.get(i)).unwrap();

        std::fs::write(
            path.join(format!("iso-{:03}.yml", num_children)),
            split_pattern_yaml,
        )
        .unwrap();
    }
}

pub fn load_split_patterns_from_dir(path: &Path, max_num_children: usize) -> SplitPatterns<2> {
    assert!(max_num_children > 1);

    let mut split_patterns: Vec<SplitPattern<2>> = Vec::new();
    for i in 0..max_num_children - 2 {
        let num_children = i + 2;

        let split_pattern_yaml =
            std::fs::read_to_string(path.join(format!("iso-{:03}.yml", num_children))).unwrap();
        let split_pattern: SplitPattern<2> = serde_yaml::from_str(&split_pattern_yaml).unwrap();
        split_pattern.assert_n_children(i);
        split_patterns.push(split_pattern);
    }
    SplitPatterns::new(split_patterns)
}*/

/* pub fn init_split_patterns(max_num_children: usize, regenerate_split_patterns: bool) -> SplitPatterns<2> {
    assert!(max_num_children > 1);

    let path = PathBuf::from("./split-patterns.yaml");

    if regenerate_split_patterns {
        let split_patterns = generate_split_patterns(max_num_children);
        write_split_patterns_to_file(&path, &split_patterns);
        split_patterns
    } else {
        load_split_patterns_from_file(&path)
    }
} */

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SceneBoundary {
    r#type: String,
    width: FT,
    height: FT,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SceneFluidBlock {
    pos: Vec<FT>,
    size: Vec<FT>,
    spacing: FT,
    volume_fill_ratio: FT,
    velocity: Vec<FT>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneConfig {
    boundary: SceneBoundary,
    blocks: Vec<SceneFluidBlock>,
}

pub fn init_fluid_sim(
    simulation_params: SimulationParams,
    scene_config: &SceneConfig,
    split_patterns: SplitPatterns<2>,
    counters_enabled: bool,
) -> FluidSimulation<DimensionUtils2d, 2> {
    // let x = 2.0 * INIT_PARTICLE_VOLUME.powf(0.5);

    let scene_center = vec2f(0., 0.);

    let mut fluid_position = Vec::new();
    let mut fluid_masses = Vec::new();
    let mut fluid_velocities = Vec::new();

    for block in &scene_config.blocks {
        add_fluid_block::<DimensionUtils2d, 2>(
            vec2f(block.pos[0], block.pos[1]),
            vec2f(block.pos[0] + block.size[0], block.pos[1] + block.size[1]),
            block.spacing,
            block.volume_fill_ratio,
            vec2f(block.velocity[0], block.velocity[1]),
            &mut fluid_position,
            &mut fluid_masses,
            &mut fluid_velocities,
        );
    }

    let num_fluid_particles = fluid_position.len();

    println!("INIT {} FLUID PARTICLES", num_fluid_particles);

    /*// ----------------------
    // init fluid particles
    let num_particles = INIT_FLUID_VOLUME / INIT_PARTICLE_VOLUME;
    let particle_box_size = (num_particles as f32).sqrt() as i32;
    let particle_spacing = INIT_GRID_SPACING;
    let mut init_particle_positions = Vec::new();
    let mut init_particle_velocities = Vec::new();
    let fluid_box_size = particle_spacing * particle_box_size as FT;
    let fluid_box_center = scene_center
        + vec2f(
            -0.4 * M,
            -target_box_width / 2. + fluid_box_size / 2. + INIT_GRID_SPACING * 2.,
        );
    let min = fluid_box_center - vec2f(fluid_box_size, fluid_box_size) * 0.5;
    for y in 0..particle_box_size {
        for x in 0..particle_box_size {
            init_particle_positions.push(vec2f(
                x as FT * particle_spacing + min.x, // - y as FT * 0.0001,
                y as FT * particle_spacing + min.y,
            ));

            let height = 1.5 * M;

            init_particle_velocities.push(vec2f(
                0.,
                -(9.81 * 2. * height).sqrt(), // m / s
            ));
        }
    }*/

    let boundary_handler: BoundaryHandler<DimensionUtils2d, 2>;

    match simulation_params.init_boundary_handler {
        InitBoundaryHandlerType::Particles => {
            // ----------------------
            // build a square of boundary particles (uniformly sampled)
            let spacing: FT = scene_config
                .blocks
                .iter()
                .map(|block| block.spacing)
                .fold(FT::INFINITY, |a, b| a.min(b));
            let num_particles_per_hedge = (scene_config.boundary.width / spacing).floor() as i32;
            let num_particles_per_vedge = (scene_config.boundary.height / spacing).floor() as i32;
            let box_width = num_particles_per_hedge as FT * spacing;
            let box_height = num_particles_per_vedge as FT * spacing;
            let boundary_min = scene_center - vec2f(box_width / 2., box_height / 2.);
            let boundary_max = scene_center + vec2f(box_width / 2., box_height / 2.);
            let mut boundary_particle_positions: Vec<V2> = Vec::new();
            for edge in [0, 1, 2, 3] {
                let (start, dir, num_particles_on_edge) = match edge {
                    0 => (
                        vec2f(boundary_min.x, boundary_min.y),
                        vec2f(spacing, 0.),
                        num_particles_per_hedge,
                    ),
                    1 => (
                        vec2f(boundary_max.x, boundary_min.y),
                        vec2f(0., spacing),
                        num_particles_per_vedge,
                    ),
                    2 => (
                        vec2f(boundary_max.x, boundary_max.y),
                        vec2f(-spacing, 0.),
                        num_particles_per_hedge,
                    ),
                    3 => (
                        vec2f(boundary_min.x, boundary_max.y),
                        vec2f(0., -spacing),
                        num_particles_per_vedge,
                    ),
                    _ => unreachable!(),
                };

                for i in 0..num_particles_on_edge {
                    boundary_particle_positions.push(start + dir * i as FT);
                }
            }

            boundary_handler =
                ParticleBasedBoundaryHandler::new(boundary_particle_positions, num_fluid_particles).into();
        }
        InitBoundaryHandlerType::AnalyticOverestimate => {
            let boundary_min =
                scene_center - vec2f(scene_config.boundary.width / 2., scene_config.boundary.height / 2.);
            let boundary_max =
                scene_center + vec2f(scene_config.boundary.width / 2., scene_config.boundary.height / 2.);

            // let sdf2d = Sdf2D::new_boundary_box(boundary_min, boundary_max);
            let sdf2d = SdfPlane::<2>::new_boundary_box(boundary_min, boundary_max);
            let sdfs = sdf2d.into_iter().map(Sdf::from).collect();

            // panic!("{:?}", sdf.probe(- vec2f(scene_config.boundary.width / 2. - 0.01, scene_config.boundary.height / 2. - 0.01)));

            boundary_handler = BoundaryWinchenbach2020::new(sdfs, num_fluid_particles).into();
        }
        InitBoundaryHandlerType::AnalyticUnderestimate => {
            let boundary_min =
                scene_center - vec2f(scene_config.boundary.width / 2., scene_config.boundary.height / 2.);
            let boundary_max =
                scene_center + vec2f(scene_config.boundary.width / 2., scene_config.boundary.height / 2.);

            let sdf2d = Sdf2D::new_boundary_box(boundary_min, boundary_max);

            boundary_handler = BoundaryWinchenbach2020::new(vec![Sdf::from(sdf2d)], num_fluid_particles).into();
        }
        InitBoundaryHandlerType::NoBoundary => {
            boundary_handler = NoBoundaryHandler {}.into();
        }
    }

    // boundary_particle_positions = Vec::new();

    // let mut rng = rand::thread_rng();
    // initial_particle_positions.shuffle(&mut rng);

    //   initial_particle_positions.push(V2F::new(40., 100.));
    //   initial_particle_positions.push(V2F::new(140., 100.));

    FluidSimulation::<DimensionUtils2d, 2>::new(
        fluid_position, //
        fluid_velocities,
        fluid_masses,
        boundary_handler,
        split_patterns,
        counters_enabled,
    )
}

pub fn init_simulation_params(simulation_params: &mut SimulationParams, scene_config: &SceneConfig) {
    {
        match PARTICLE_SIZES {
            ParticleSizes::Adaptive => {
                // this attribute is not used for adaptive simulations
                simulation_params.h = 0.;
                println!("set h to 0");
            }
            ParticleSizes::Uniform => {
                let v = scene_config.blocks[0].spacing
                    * scene_config.blocks[0].spacing
                    * scene_config.blocks[0].volume_fill_ratio;

                simulation_params.h = ETA * <DimensionUtils2d as DimensionUtils<2>>::sphere_volume_to_radius(v);
                println!(
                    "h:{} spacing:{} h/spacing:{}",
                    simulation_params.h,
                    INIT_GRID_SPACING,
                    simulation_params.h / INIT_GRID_SPACING
                );
            }
        }
    }
}

pub fn is_ft_approx_eq<FT: Float>(a: FT, b: FT, tolerance: FT) -> bool {
    assert!(!a.is_nan());
    assert!(!b.is_nan());
    b <= a + tolerance && b >= a - tolerance
}

pub fn assert_ft_approx_eq2<FT: Float + Display>(
    a: FT,
    b: FT,
    tolerance: FT,
    s: impl FnOnce() -> (String, String, String),
) {
    if !is_ft_approx_eq(a, b, tolerance) {
        let (desc, astr, bstr) = s();
        panic!(
            "{} value not equal with a tolerance of {}:\n\t{}={}\n\t{}={}\n",
            desc, tolerance, astr, a, bstr, b
        );
    }
}

pub fn write_statistics(fluid_simulation: &FluidSimulation<DimensionUtils2d, 2>) -> String {
    let mut s = String::new();

    let simulation_time = fluid_simulation
        .pcounters
        .counters
        .get("simulation-step")
        .unwrap()
        .sum()
        .as_secs_f64();

    let avg_particle_count = fluid_simulation.vcounters.counters.get("particle-count").unwrap().avg();

    let avg_div_free_iter = fluid_simulation.vcounters.counters.get("div-iterations").unwrap().avg();

    let avg_density_iter = fluid_simulation
        .vcounters
        .counters
        .get("density-iterations")
        .unwrap()
        .avg();

    writeln!(
        s,
        "${:.2}\\si{{\\second}}$ & {} & {:.02} & {:.02} & - \\\\",
        simulation_time,
        avg_particle_count.round() as i32,
        avg_div_free_iter,
        avg_density_iter,
    )
    .unwrap();
    writeln!(s).unwrap();

    writeln!(
        s,
        "simulation-time: {}ms",
        fluid_simulation
            .pcounters
            .counters
            .get("simulation-step")
            .unwrap()
            .sum()
            .as_secs_f64()
            * 1000.
    )
    .unwrap();
    writeln!(s).unwrap();

    let mut v = fluid_simulation
        .pcounters
        .counters
        .clone()
        .into_iter()
        .collect::<Vec<_>>();
    v.sort_by(|x, y| x.0.cmp(&y.0));
    for (label, pcounter) in v {
        writeln!(s, "{}: avg:{}ms", label, pcounter.avg().as_secs_f64() * 1000.).unwrap();
    }
    writeln!(s).unwrap();

    let mut v = fluid_simulation
        .vcounters
        .counters
        .clone()
        .into_iter()
        .collect::<Vec<_>>();
    v.sort_by(|x, y| x.0.cmp(&y.0));
    for (label, pcounter) in v {
        writeln!(
            s,
            "{}: min:{} max:{} avg:{}",
            label,
            pcounter.min(),
            pcounter.max(),
            pcounter.avg()
        )
        .unwrap();
    }

    s
}

pub fn assert_ft_approx_eq<FT: Float + Display>(a: FT, b: FT, tolerance: FT, s: impl FnOnce() -> String) {
    if !is_ft_approx_eq(a, b, tolerance) {
        panic!(
            "{} value not equal with a tolerance of {}:\n\ta={}\n\tb={}\n",
            s(),
            tolerance,
            a,
            b
        );
    }
}
