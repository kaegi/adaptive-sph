
/*

Talking points:
    SPH literature for gather, scatter and symmteric formulation
    symmtrizing neighborhood relationship was developed from ime (how to cite)
    not tested
    (https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.14090)


https://github.com/InteractiveComputerGraphics/blender-sequence-loader/

How to delete paricles from array in paralell?

TODO:
    Fixed masses but different masses
    Change neighborhood lists to Vec<Vec<ParticleId>>
        -> Need to wipe neighborhood lists after ne&ighborhood size changes???

TODO:
    Derive a_ii for semi-analytic boundary handling.
    Find cause of boundary penetration for semi-analytic boundary handling.
    Implement adaptive particle sizes.
    Dummy DimensionUtils
    Unsafe ThreadSafePtr::new?

NEXT STEPS:
    -Think about: Is Akinci boundary handling possible with large fluid particles/small boundary particles?
    -Implement numerical test for a_ii by evaluting matrix (a_{ij} * p_j) for p_i=1 and all other p_x=0

OPEN QUESTIONS:
    Adaptiation for particles with boundary accelerations. Do I need to handle strongly coupled objects?
    Inconsistency: IISPH or DFSPH, Jacobi-Style or "Simplified". Which underlying acceleration formula in the derivation?
    Winchenbach2020 + adaptivity: Omega_i scaling factors in a_ii and the force terms (especially the rigid->fluid acceleration)?

    Winchenbach2020 (boundary handling) after Eq. 41: "For boundary objects we assume that p_b is equal to the object’s (FLUID????) respective rest density"


Insight:
    -smaller particles
        -> for same viscosity coefficient smaller timesteps needed for stable simulation (explicit viscosity model)
        -> smaller timestep from CFL condition

Pressure gradient formulation differs in IISPH/DFSPH for fluid-boundary interaction ("simple" SPH gradient) and fluid-fluid particles (symmetric "SPH" gradient).
"MLS Pressure Boundaries" do not include the self-contributions in the diagonal element (e.g p_i changes p_b which will further change a_i).

*
* Fragen:
*  SPH-Interpolation vom Distanz-Feld mit "Rest-Density" oder SPH-Density? [phi_i = SUM(m_j/p_j * phi_j)]
*/

/*
   Notes for openMaelstrom:
       arrays.volume.first[i] is the same as "mass[i] / rest_density" in adaptive-SPH
       arrays.radius          is "r_base"
       position[i].val.w      is the same as "h[i]" (or H[i]) in adaptive-sph
*/

/**
 * Found bugs:
 *   - boundary term was only added in density_adv (not in density)
 *   - the boundary formula used velocity_adv instead of velocity
 */

mod platform;
mod simulation;

pub use simulation::*;

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    platform::start();
}
