use crate::{floating_type_mod::FT, sph_kernels::DimensionUtils, VF};
use serde::{Deserialize, Serialize};

#[derive(PartialEq, Eq, Debug, Clone, Copy, Serialize, Deserialize)]
pub enum FillStashWith {
    SurfaceDistanceFirstIteration,
    SurfaceDistanceMiddle,
}

#[derive(PartialEq, Eq, Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SizingFunction {
    Radius2,
    Radius,
    Mass,
}

#[derive(PartialEq, Eq, Debug, Clone, Copy, Serialize, Deserialize)]
pub enum BoundaryPenaltyTerm {
    None,
    Linear,
    Quadratic1,
    Quadratic2,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SimulationParams {
    pub rest_density: FT,
    pub cfl_factor: FT,
    pub max_dt: FT,
    pub h: FT,
    pub use_iisph: bool,
    pub viscosity: FT,
    pub viscosity_type: ViscosityType,
    pub gravity: FT,
    pub check_aii: bool,

    // parameters for level estimation
    pub level_estimation_method: LevelEstimationMethod,
    // given in "number of particle radiuses"
    pub maximum_range: FT,

    // only used for IISPH
    pub jacobi_omega: FT,

    // only used for WCSPH
    pub eos_stiffness: FT,
    pub eos_power: i32,

    pub neighborhood_search_algorithm: NeighborhoodSearchAlgorithm,
    pub init_boundary_handler: InitBoundaryHandlerType,
    pub support_length_estimation: SupportLengthEstimation,

    // only used with BoundaryHandlerType::Analytic for the SDF gradient computation
    pub sdf_gradient_eps: FT,

    pub fail_on_missing_split_pattern: bool,
    pub pull_fluid_to: Option<VF<3>>,

    // -------------------
    // ADAPTIVE SIMULTION ONLY

    // enforce an upper limit on the neighbor count?
    pub constrain_neighborhood_count: bool,
    pub particle_radius_fine: FT,
    pub particle_radius_base: FT,
    // this value is positive even though it is meant as a measure for "how much inside the fluid" until the "base particle radius" is used
    pub maximum_surface_distance: FT,
    pub minimum_share_partners: u16,
    pub minimum_merge_partners: u16,
    pub merging: bool,
    pub sharing: bool,
    pub splitting: bool,
    pub max_mass_transfer_sharing: FT,
    pub max_mass_transfer_merging: FT,
    pub max_share_distance: FT,
    pub max_merge_distance: FT,
    pub allow_merge_with_optimal_particle: bool,
    pub allow_share_with_optimal_particle: bool,
    pub allow_share_with_too_small_particle: bool,
    pub allow_merge_on_size_difference: bool,

    pub boundary_is_fluid_surface: bool,
    pub use_extended_range_for_level_estimation: bool,

    pub pressure_solver_method: PressureSolverMethod,
    pub iisph_max_avg_density_error: FT,
    pub hybrid_dfsph_factor: FT,
    pub hybrid_dfsph_max_avg_density_error: FT,
    pub hybrid_dfsph_max_avg_divergence_error: FT,
    pub hybrid_dfsph_density_source_term: HybridDfsphDensitySourceTerm,
    pub hybrid_dfsph_non_pressure_accel_before_divergence_free: bool,

    pub check_neighborhood: bool,

    pub fill_stash_with: Option<FillStashWith>,

    pub boundary_penalty_term: BoundaryPenaltyTerm,

    pub sizing_function: SizingFunction,

    pub level_estimation_after_advection: bool,
    pub level_estimation_range: FT,

    pub operator_discretization: OperatorDiscretization,
    pub operator_discretization_for_diagonal: Option<OperatorDiscretization>,

    pub max_iters: usize,
}

#[derive(PartialEq, Eq, Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OperatorDiscretization {
    // this uses the simple SPH gradient for fluid-boundary interacitons
    // boundary accelerations is "-sum_b[m_b * p_b / rho_b^2 * weight_grad_i]" (used in Akinci2021, IISPH, DFSPH)
    ConsistentSimpleGradient,

    // boundary accelerations is "-sum_b[m_b * (p_i / rho_i^2 + p_b / rho_b^2) * weight_grad_i]"
    // with p_b = p_i and rho_b = p_0
    // used in winchenbach
    ConsistentSymmetricGradient,

    Winchenbach2020,
}

impl SimulationParams {
    pub fn mass_fine<DU: DimensionUtils<D>, const D: usize>(&self) -> FT {
        DU::radius_to_sphere_volume(self.particle_radius_fine) * self.rest_density
    }

    pub fn mass_base<DU: DimensionUtils<D>, const D: usize>(&self) -> FT {
        DU::radius_to_sphere_volume(self.particle_radius_base) * self.rest_density
    }

    pub fn gravity_vector<const D: usize>(&self) -> VF<D> {
        let mut data: [FT; D] = [0.; D];
        if D == 2 {
            data[0] = 0.;
            data[1] = self.gravity;
        } else {
            data[0] = 0.;
            data[1] = self.gravity;
            data[2] = 0.;
        }

        VF::<D>::from_column_slice(&data)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ViscosityType {
    WCSPH,
    ApproxLaplace,
    XSPH,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NeighborhoodSearchAlgorithm {
    // "Grid" only works for constant kernel radius
    Grid,
    RStar,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InitBoundaryHandlerType {
    Particles,
    AnalyticUnderestimate,
    AnalyticOverestimate,
    NoBoundary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SupportLengthEstimation {
    // "Constrained Neighbor Lists for SPH-based Fluid Simulations" Eq. 4
    FromDistribution,
    FromDistributionClamped1,
    FromDistributionClamped2,

    // "Constrained Neighbor Lists for SPH-based Fluid Simulations" Eq. 4 (Adapted)
    FromDistribution2,

    // Jens Orthmann and Andreas Kolb "Temporal Blending for Adaptive SPH" Eq. 5
    FromMass,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LevelEstimationMethod {
    // Do not perform level estimation
    None,

    // 2013 Mass Preserving Multi-Scale SPH, Christopher Jon Horvath et al. Eq 10 & 11
    //  --> surface detection not working to well with adaptive particles
    CenterDiff,

    // Source code of openMaelstrom "surfaceDetection.cu"
    EmptyAngle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PressureSolverMethod {
    IISPH,
    IISPH2,

    /// small timesteps: only correct positions with density solvee
    /// large tiemsteps: correct positions & velocities with density solve
    HybridDFSPH,

    OnlyDivergence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HybridDfsphDensitySourceTerm {
    DensityAndDivergence,
    OnlyDensity,
}

/*
    [label = "Rest Density (1/100)", min=0, max=400, scale=1./100., cast=f32]
    [label = "Time steps in ms", min=1, max=80, scale=1./1000., cast=f32]
    [label = "Kernel Support Radius (1/1000m or 1mm)", min=0, max=100, scale=1./1000., cast=f32]
    [label = "Use IISPH", min=0, max=1, scale=1, cast=i32]
    [label = "Viscosity (1/10000)", min=0, max=10000, scale=1./10000., cast=f32]
    [label = "Viscosity Type (0=WCSPH, 1=Approx Laplace, 2=XSPH)", min=0, max=2, scale=1, cast=i32]
    [label = "EOS Stiffness", min=0, max=1000, scale=1., cast=f32]
    [label = "EOS Power", min=0, max=20, scale=1, cast=i32]
    [label = "Jacobi Omega (1/1000)", min=0, max=500, scale=1./1000., cast=f32]
    [label = "Gravity", min=0, max=500, scale=1./1000., cast=f32]
    [label = "Check a_ii"]
*/
