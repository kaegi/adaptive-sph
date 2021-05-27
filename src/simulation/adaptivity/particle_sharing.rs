use crate::{
    adaptivity::{ParticleSizeClass, MERGE_PARTNER_AVAILABLE, MERGE_PARTNER_DELETE, PARTICLE_SIZE_FACTOR_LARGE},
    boundary_handler::BoundaryHandler,
    concurrency::{par_iter_mut1, par_iter_mut2, par_iter_mut3},
    floating_type_mod::FT,
    local_smoothing_length_from_mass,
    neighborhood_search::NeighborhoodCache,
    simulation_parameters::SimulationParams,
    sph_kernels::{smoothing_length, DimensionUtils},
    thread_safe_ptr::ThreadSafeMutPtr,
    LevelEstimationState, ParticleVec,
};

pub fn find_share_partner_sequential<DU: DimensionUtils<D>, const D: usize>(
    particles: &mut ParticleVec<D>,
    neighs: &NeighborhoodCache,
    simulation_params: SimulationParams,
    dt: FT,
) {
    par_iter_mut1(&mut particles.merge_partner, |_, p_merge_partner| {
        *p_merge_partner.get_mut() = MERGE_PARTNER_AVAILABLE;
    });

    let merge_partner = &mut particles.merge_partner;
    let particle_size_class = &particles.particle_size_class;
    let mass = &particles.mass;
    let level_estimation = &particles.level_estimation;
    let merge_counter = &mut particles.merge_counter;
    let position = &particles.position;
    let h = &particles.h2;

    let mut num_shared_particles = 0;

    for i in 0..particles.position.len() {
        // println!("{}", i);
        merge_counter[i] = 0;

        // only share "S" particles into neighbors
        if particles.particle_size_class[i] != ParticleSizeClass::Large {
            continue;
        }

        for j in neighs.iter(i) {
            if i == j {
                continue;
            }

            // TODO: introduce option "allow/disallow share with Optimal particles"
            // do not distribute mass to particles that are already to large
            let can_share_to_j = match particle_size_class[j] {
                ParticleSizeClass::Small => true,
                ParticleSizeClass::TooSmall => simulation_params.allow_share_with_too_small_particle,
                ParticleSizeClass::Large | ParticleSizeClass::TooLarge => false,
                ParticleSizeClass::Optimal => simulation_params.allow_share_with_optimal_particle,
            };
            if !can_share_to_j {
                continue;
            }

            // XXX: VERY IMPORTANT long distance shares lead to popping/unstable behavior
            let xij = position[i] - position[j];
            let max_dist = smoothing_length(h, i, j, simulation_params) * simulation_params.max_share_distance;
            if xij.norm_squared() > max_dist * max_dist {
                continue;
            }

            let dropped_mass_i = dropped_mass_sharing::<DU, D>(&level_estimation[i], mass[i], dt, simulation_params);

            let new_mass_j = mass[j] + dropped_mass_i / (merge_counter[i] + 1) as FT;
            let target_mass_j = level_estimation[j].target_mass::<DU, D>(simulation_params);
            if new_mass_j >= target_mass_j * PARTICLE_SIZE_FACTOR_LARGE {
                continue;
            }
            if new_mass_j > simulation_params.mass_base::<DU, D>() {
                continue;
            }

            if *merge_partner[j].get_mut() != MERGE_PARTNER_AVAILABLE {
                // the current particle can not be shared into other particles,
                // since the neighboring particle is being used as a share partner
                continue;
            }

            if merge_counter[i] == 0 {
                if *merge_partner[i].get_mut() != MERGE_PARTNER_AVAILABLE {
                    // the current particle can not be shared into other particles,
                    // since this particle is being used as a share partner
                    continue;
                }
                *merge_partner[i].get_mut() = MERGE_PARTNER_DELETE;
            }

            *merge_partner[j].get_mut() = u32::try_from(i).unwrap();

            merge_counter[i] += 1;
            num_shared_particles += 1;

            assert!(merge_counter[i] < 1000);
        }

        // if merge_counter[i] > 0 {
        //     break;
        // }
    }

    if true {
        validate_share_partners::<DU, D>(particles, neighs) as isize;
    }

    println!("SEQUENTIAL SHARE {} shares", num_shared_particles);
}

pub fn validate_share_partners<DU: DimensionUtils<D>, const D: usize>(
    particles: &mut ParticleVec<D>,
    neighs: &NeighborhoodCache,
) -> usize {
    // validate share partners
    let mut num_shared_particles = 0;

    let num_fluid_particles = particles.merge_counter.len();

    for i in 0..num_fluid_particles {
        if particles.merge_counter[i] > 0 {
            assert!(particles.particle_size_class[i] == ParticleSizeClass::Large);
            num_shared_particles += 1;

            assert!(*particles.merge_partner[i].get_mut() == MERGE_PARTNER_DELETE);

            // neighbors
            let mut merge_counter2 = 0;
            for j in neighs.iter(i) {
                if *particles.merge_partner[j].get_mut() == i as u32 {
                    merge_counter2 += 1;
                }
            }

            assert!(particles.merge_counter[i] == merge_counter2);
        } else {
            assert!(*particles.merge_partner[i].get_mut() != MERGE_PARTNER_DELETE);

            if *particles.merge_partner[i].get_mut() != MERGE_PARTNER_AVAILABLE {
                // has a share partner
                let this_merge_partner = *particles.merge_partner[i].get_mut() as usize;
                assert!(*particles.merge_partner[this_merge_partner].get_mut() == MERGE_PARTNER_DELETE);
            }
        }
    }

    num_shared_particles
}

pub fn share_particles<DU: DimensionUtils<D>, const D: usize>(
    particles: &mut ParticleVec<D>,
    _neighs: &mut NeighborhoodCache,
    _boundary_handler: &mut BoundaryHandler<DU, D>,
    simulation_params: SimulationParams,
    dt: FT,
) {
    let position_ptr = ThreadSafeMutPtr::new(particles.position.as_mut_ptr());
    let mass_ptr = ThreadSafeMutPtr::new(particles.mass.as_mut_ptr());
    let velocity_ptr = ThreadSafeMutPtr::new(particles.velocity.as_mut_ptr());

    // share particles
    par_iter_mut2(
        &mut particles.merge_partner,
        &mut particles.h2_next,
        |i, p_merge_partner, p_h_next| {
            match *p_merge_partner.get_mut() {
                MERGE_PARTNER_AVAILABLE => { /* this particle is neigher getting nor distributing mass */ }
                MERGE_PARTNER_DELETE => { /* this particle is distributing mass */ }
                j => {
                    let j = j as usize;

                    if particles.merge_counter[j] < simulation_params.minimum_share_partners {
                        return;
                    }

                    // !!! merge_partner[j] is guaranteed to be MERGE_PARTNER_DELETE and therefore
                    // no attribute is modified for j concurrently !!!

                    unsafe {
                        let position_i = &mut *position_ptr.offset(i as isize);
                        let position_j = *position_ptr.offset(j as isize);

                        let mass_i_mut = &mut *mass_ptr.offset(i as isize);
                        let mass_i = *mass_i_mut;
                        let mass_j = *mass_ptr.offset(j as isize);

                        let velocity_i = &mut *velocity_ptr.offset(i as isize);
                        let velocity_j = *velocity_ptr.offset(j as isize);

                        let dropped_mass_j = dropped_mass_sharing::<DU, D>(
                            &particles.level_estimation[j],
                            mass_j,
                            dt,
                            simulation_params,
                        );

                        let mass_n = dropped_mass_j / particles.merge_counter[j] as FT;
                        let mass = mass_i + mass_n;

                        *velocity_i = (mass_i * *velocity_i + mass_n * velocity_j) / mass;
                        *position_i = (mass_i * *position_i + mass_n * position_j) / mass;
                        *mass_i_mut = mass;

                        *p_h_next = local_smoothing_length_from_mass::<DU, D>(mass, simulation_params.rest_density);
                    }
                }
            }
        },
    );

    par_iter_mut3(
        &mut particles.merge_partner,
        &mut particles.h2_next,
        &mut particles.mass,
        |i, p_merge_partner, p_h_next, p_mass| {
            if *p_merge_partner.get_mut() != MERGE_PARTNER_DELETE {
                return;
            }

            if particles.merge_counter[i] < simulation_params.minimum_share_partners {
                return;
            }

            // let target_mass = particles.level_estimation[i].target_mass::<DU, D>(simulation_params);
            let dropped_mass =
                dropped_mass_sharing::<DU, D>(&particles.level_estimation[i], *p_mass, dt, simulation_params);
            // println!(
            //     "{} dropped {} of {} ({:.2})",
            //     i,
            //     dropped_mass,
            //     *p_mass - target_mass,
            //     dropped_mass / (*p_mass - target_mass) * 100.
            // );
            *p_mass -= dropped_mass;
            *p_h_next = local_smoothing_length_from_mass::<DU, D>(*p_mass, simulation_params.rest_density);
        },
    );
}

fn dropped_mass_sharing<DU: DimensionUtils<D>, const D: usize>(
    level_estimation: &LevelEstimationState,
    mass: FT,
    dt: FT,
    simulation_params: SimulationParams,
) -> FT {
    let target_mass = level_estimation.target_mass::<DU, D>(simulation_params);
    FT::min(
        mass - target_mass,
        target_mass * simulation_params.max_mass_transfer_sharing * dt,
    )
}
