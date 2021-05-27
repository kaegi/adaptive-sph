
pub mod adaptivity;
pub mod boundary_handler;
pub mod color_map;
pub mod colors;
pub mod concurrency;
pub mod neighborhood_search;
pub mod properties_window;
pub mod sdf;
pub mod simulation_parameters;
pub mod sph_kernels;
#[allow(dead_code)]
pub mod thread_safe_ptr;
pub mod simulation;

pub type IT = i32;

#[cfg(feature = "double-precision")]
pub mod floating_type_mod {
    pub type FT = f64;
    pub use std::f64::consts::{FRAC_1_PI, PI, TAU};
}

#[cfg(not(feature = "double-precision"))]
pub mod floating_type_mod {
    pub type FT = f32;
    pub use std::f32::consts::{FRAC_1_PI, PI, TAU};
}

use floating_type_mod::FT;

use nalgebra::{SVector};

#[allow(dead_code)]
pub type V<FT, const D: usize> = SVector<FT, D>;
// #[allow(dead_code)]
// pub type M<FT, const D: usize> = SMatrix<FT, D, D>;

pub type VF<const D: usize> = V<FT, D>;
pub type VI<const D: usize> = V<IT, D>;

pub type V2 = V<FT, 2>;
// #[allow(dead_code)]
// pub type M2 = M<FT, 2>;

pub type V3 = V<FT, 3>;

#[allow(dead_code)]
pub type V2F = V<f32, 2>;
#[allow(dead_code)]
pub type V2D = V<f64, 2>;
#[allow(dead_code)]
pub type V2I = V<i32, 2>;

pub fn vec2u(x: u32, y: u32) -> V<u32, 2> {
    [x, y].into()
}

pub fn vec2f(x: FT, y: FT) -> V<FT, 2> {
    [x, y].into()
}

pub fn vec3f(x: FT, y: FT, z: FT) -> V<FT, 3> {
    [x, y, z].into()
}

pub use simulation::*;