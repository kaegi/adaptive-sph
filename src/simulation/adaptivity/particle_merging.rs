use std::sync::atomic::Ordering;

use crate::{
    adaptivity::{ParticleSizeClass, MERGE_PARTNER_AVAILABLE, MERGE_PARTNER_DELETE, PARTICLE_SIZE_FACTOR_LARGE},
    boundary_handler::{BoundaryHandler, BoundaryHandlerTrait},
    concurrency::{par_iter_mut1, par_iter_mut2},
    local_smoothing_length_from_mass,
    neighborhood_search::NeighborhoodCache,
    simulation_parameters::SimulationParams,
    sph_kernels::{smoothing_length, DimensionUtils},
    thread_safe_ptr::ThreadSafeMutPtr,
    LevelEstimationState, ParticleVec, floating_type_mod::FT,
};

pub fn find_merge_partner_sequential<DU: DimensionUtils<D>, const D: usize>(
    particles: &mut ParticleVec<D>,
    neighs: &NeighborhoodCache,
    dt: FT,
    simulation_params: SimulationParams,
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

    let mut num_merged_particles = 0;

    // let mut sorted_indices: Vec<usize> = (0..particles.position.len()).collect();
    // sorted_indices.sort_unstable_by(|i, j| FT::partial_cmp(&particles.mass[*i], &particles.mass[*j]).unwrap());

    // for i in 0..particles.position.len() - 1 {
    //     assert!(particles.mass[sorted_indices[i]] <= particles.mass[sorted_indices[i + 1]]);
    // }

    for i in 0..particles.position.len() {
        // println!("{}", i);
        merge_counter[i] = 0;

        // only merge "S" particles into neighbors
        if particles.particle_size_class[i] != ParticleSizeClass::TooSmall {
            continue;
        }

        for j in neighs.iter(i) {
            if i == j {
                continue;
            }

            let mut can_merge_to_j = match particle_size_class[j] {
                ParticleSizeClass::Large | ParticleSizeClass::TooLarge => false,
                ParticleSizeClass::Optimal => simulation_params.allow_merge_with_optimal_particle,
                ParticleSizeClass::Small | ParticleSizeClass::TooSmall => true,
            };

            if simulation_params.allow_merge_on_size_difference && particles.mass[j] > 5. * particles.mass[i] {
                can_merge_to_j = true;
            }

            if !can_merge_to_j {
                continue;
            }

            // XXX: VERY IMPORTANT long distance merges lead to popping/unstable behavior
            let xij = position[i] - position[j];
            let max_dist = smoothing_length(h, i, j, simulation_params) * simulation_params.max_merge_distance;
            if xij.norm_squared() > max_dist * max_dist {
                continue;
            }

            let dropped_mass =
                dropped_mass_merging::<DU, D>(&particles.level_estimation[i], mass[i], dt, simulation_params);
            let merged_mass = mass[j] + dropped_mass / (merge_counter[i] + 1) as FT;
            let target_mass = level_estimation[j].target_mass::<DU, D>(simulation_params);
            if merged_mass >= target_mass * PARTICLE_SIZE_FACTOR_LARGE {
                continue;
            }
            if merged_mass > simulation_params.mass_base::<DU, D>() {
                continue;
            }

            if *merge_partner[j].get_mut() != MERGE_PARTNER_AVAILABLE {
                // the current particle can not be merged into other particles,
                // since the neighboring particle is being used as a merge partner
                continue;
            }

            if merge_counter[i] == 0 {
                if *merge_partner[i].get_mut() != MERGE_PARTNER_AVAILABLE {
                    // the current particle can not be merged into other particles,
                    // since this particle is being used as a merge partner
                    continue;
                }
                *merge_partner[i].get_mut() = MERGE_PARTNER_DELETE;
            }

            *merge_partner[j].get_mut() = u32::try_from(i).unwrap();

            merge_counter[i] += 1;
            num_merged_particles += 1;

            assert!(merge_counter[i] < 1000);
        }

        // if merge_counter[i] > 0 {
        //     break;
        // }
    }

    if true {
        validate_merge_partners::<DU, D>(particles, neighs, simulation_params) as isize;
    }

    println!("SEQUENTIAL MERGE {} merges", num_merged_particles);
}

/*pub fn find_merge_partner_parallel<DU: DimensionUtils<D>, const D: usize>(
    particles: &mut ParticleVec<D>,
    neighs: &NeighborhoodCache,
    simulation_params: SimulationParams,
) {
    par_iter_mut1(&mut particles.merge_partner, |_, p_merge_partner| {
        *p_merge_partner.get_mut() = MERGE_PARTNER_AVAILABLE;
    });

    let critical_locking = AtomicI32::new(0);
    let merge_partner = &particles.merge_partner;
    let particle_size_class = &particles.particle_size_class;
    let mass = &particles.mass;
    let level_estimation = &particles.level_estimation;

    // find merge partners (see "Infinite Continuous Adaptivity for Incompressible SPH" section 6.2)
    par_iter_mut1(&mut particles.merge_counter, |i, p_merge_counter| {
        // only merge "S" particles into neighbors
        if particles.particle_size_class[i] != ParticleSizeClass::TooSmall {
            *p_merge_counter = 0;
            return;
        }

        let memory_ordering = Ordering::SeqCst;
        // let memory_ordering = Ordering::Relaxed; // not sure whether this is safe

        // TODO: sort neighbors from near to far -> does that improve stability (because of smaller distance mass trasfers?)?
        let mut merge_counter = 0;
        for j in neighs.iter(i) {
            if i == j {
                continue;
            }

            // TODO: introduce option "allow/disallow merge with Optimal particles"
            // do not distribute mass to particles that are already to large
            match particle_size_class[j] {
                ParticleSizeClass::Large | ParticleSizeClass::TooLarge => {
                    continue;
                }
                _ => {}
            }

            // TODO: limit distance to which mass is distributed (see Winchenbach2017 and 'Merging.cuh')

            // XXX: VERY IMPORTANT long distance merges lead to popping/unstable behavior
            let merged_mass = mass[j] + mass[i] / (merge_counter + 1) as FT;
            let target_mass = level_estimation[j].target_mass::<DU, D>(simulation_params);
            if merged_mass >= target_mass * PARTICLE_SIZE_FACTOR_LARGE {
                continue;
            }

            if let Err(_) = merge_partner[j].compare_exchange(
                MERGE_PARTNER_AVAILABLE,
                u32::try_from(i).unwrap(),
                memory_ordering,
                memory_ordering,
            ) {
                // the current particle can not be merged into other particles,
                // since is (probably) being used as a merge partner
                continue;
            }

            if merge_counter == 0 {
                if let Err(_) = merge_partner[i].compare_exchange(
                    MERGE_PARTNER_AVAILABLE,
                    MERGE_PARTNER_DELETE,
                    memory_ordering,
                    memory_ordering,
                ) {
                    // the current particle can not be merged into other particles,
                    // since is (probably) being used as a merge partner

                    // release first merge-partner (undo whole transaction)
                    merge_partner[j].store(MERGE_PARTNER_AVAILABLE, Ordering::SeqCst);

                    // if this happens often, there might be many particles that were
                    // not merged just because of the (safe) race conditions in this
                    // code -> count the number of time this happens to estimate the severity
                    critical_locking.fetch_add(1, Ordering::SeqCst);

                    break;
                }
            }

            assert!(merge_counter < 1000);
            merge_counter += 1;
        }

        // particles.merge_partner[i]
        *p_merge_counter = merge_counter;
    });

    let num_merged_particles: isize;
    if true {
        num_merged_particles = validate_merge_partners::<DU, D>(particles, neighs, simulation_params) as isize;
    } else {
        num_merged_particles = -1;
    }

    println!(
        "CRITICAL LOCKING happened {} times, {} merges",
        critical_locking.into_inner(),
        num_merged_particles
    );
}*/

pub fn validate_merge_partners<DU: DimensionUtils<D>, const D: usize>(
    particles: &mut ParticleVec<D>,
    neighs: &NeighborhoodCache,
    _simulation_params: SimulationParams,
) -> usize {
    // validate merge partners
    let mut num_merged_particles = 0;

    let num_fluid_particles = particles.merge_counter.len();

    for i in 0..num_fluid_particles {
        if particles.merge_counter[i] > 0 {
            assert!(particles.particle_size_class[i] == ParticleSizeClass::TooSmall);
            num_merged_particles += 1;

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
                // has a merge partner
                let this_merge_partner = *particles.merge_partner[i].get_mut() as usize;
                assert!(*particles.merge_partner[this_merge_partner].get_mut() == MERGE_PARTNER_DELETE);
            }
        }
    }

    num_merged_particles
}

pub fn merge_particles<DU: DimensionUtils<D>, const D: usize>(
    particles: &mut ParticleVec<D>,
    neighs: &mut NeighborhoodCache,
    boundary_handler: &mut BoundaryHandler<DU, D>,
    dt: FT,
    simulation_params: SimulationParams,
) {
    let position_ptr = ThreadSafeMutPtr::new(particles.position.as_mut_ptr());
    let mass_ptr = ThreadSafeMutPtr::new(particles.mass.as_mut_ptr());
    let velocity_ptr = ThreadSafeMutPtr::new(particles.velocity.as_mut_ptr());

    // merge particles
    par_iter_mut2(
        &mut particles.merge_partner,
        &mut particles.h2_next,
        |i, p_merge_partner, p_h_next| {
            match p_merge_partner.load(Ordering::Relaxed) {
                MERGE_PARTNER_AVAILABLE => { /* this particle is neigher getting nor distributing mass */ }
                MERGE_PARTNER_DELETE => { /* this particle is distributing mass */ }
                j => {
                    let j = j as usize;

                    // !!! merge_partner[j] is guaranteed to be MERGE_PARTNER_DELETE and therefore
                    // not attribute is modified for j concurrently !!!

                    if particles.merge_counter[j] < simulation_params.minimum_merge_partners {
                        return;
                    }

                    unsafe {
                        let position_i = &mut *position_ptr.offset(i as isize);
                        let position_j = *position_ptr.offset(j as isize);

                        let mass_i_mut = &mut *mass_ptr.offset(i as isize);
                        let mass_i = *mass_i_mut;
                        let mass_j = *mass_ptr.offset(j as isize);
                        let dropped_mass = dropped_mass_merging::<DU, D>(
                            &particles.level_estimation[j],
                            mass_j,
                            dt,
                            simulation_params,
                        );

                        let velocity_i = &mut *velocity_ptr.offset(i as isize);
                        let velocity_j = *velocity_ptr.offset(j as isize);

                        let mass_n = dropped_mass / particles.merge_counter[j] as FT;
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

    // let total_mass: FT = particles
    //     .mass
    //     .iter()
    //     .zip(particles.merge_counter.iter())
    //     .filter(|(x, i)| **i == 0)
    //     .map(|(x, i)| *x)
    //     .sum();
    // println!("total mass {}", total_mass);

    let num_fluid_particles = particles.position.len();

    // delete particles by swapping them to the end of the array
    let mut last_particle_id = num_fluid_particles - 1;
    let mut i = 0;
    loop {
        if i > last_particle_id {
            break;
        }
        if *particles.merge_partner[i].get_mut() == MERGE_PARTNER_DELETE
            && particles.merge_counter[i] >= simulation_params.minimum_merge_partners
        {
            // println!("particles.mass[i]={}", particles.mass[i]);
            let dropped_mass =
                dropped_mass_merging::<DU, D>(&particles.level_estimation[i], particles.mass[i], dt, simulation_params);
            particles.mass[i] -= dropped_mass;

            if particles.mass[i] < 0.000001 {
                particles.swap(i, last_particle_id);
                neighs.swap(i, last_particle_id);
                boundary_handler.swap(i, last_particle_id);
                last_particle_id -= 1;
                continue;
            }
        }

        i += 1;
    }

    particles.truncate(last_particle_id + 1);
    neighs.truncate(last_particle_id + 1);
    boundary_handler.truncate(last_particle_id + 1);
}

fn dropped_mass_merging<DU: DimensionUtils<D>, const D: usize>(
    level_estimation: &LevelEstimationState,
    mass: FT,
    dt: FT,
    simulation_params: SimulationParams,
) -> FT {
    if false {
        let target_mass = level_estimation.target_mass::<DU, D>(simulation_params);
        FT::min(mass, target_mass * simulation_params.max_mass_transfer_merging * dt)
    } else {
        mass
    }
}
