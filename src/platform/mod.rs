#[cfg(not(target_arch = "wasm32"))]
mod desktop;

#[cfg(not(target_arch = "wasm32"))]
pub use desktop::start;

#[cfg(target_arch = "wasm32")]
pub mod web;
