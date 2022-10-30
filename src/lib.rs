/*!
This lib.rs file is used when compiling for target `wasm32-unknown-unknown`
*/

mod platform;
mod simulation;

pub use simulation::*;

/// Avoids 'unused' warnings.
#[cfg(not(target_arch = "wasm32"))]
pub use platform::start;
