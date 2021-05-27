use std::{fs::create_dir_all, iter::repeat, mem::swap};

use rand::Rng;
use serde::{Deserialize, Serialize};
use svg::{node::element::Circle, Document};

use crate::{
    boundary_handler::{BoundaryHandler, BoundaryHandlerTrait},
    floating_type_mod::{FT, TAU},
    local_smoothing_length_from_mass, local_smoothing_length_from_volume,
    neighborhood_search::NeighborhoodCache,
    simulation_parameters::SimulationParams,
    sph_kernels::DimensionUtils,
    ParticleVec, INIT_REST_DENSITY, VF,
};

use super::ParticleSizeClass;

pub fn split_particles<DU: DimensionUtils<D>, const D: usize>(
    particles: &mut ParticleVec<D>,
    neighs: &mut NeighborhoodCache,
    boundary_handler: &mut BoundaryHandler<DU, D>,
    simulation_params: SimulationParams,
    split_patterns: &SplitPatterns<D>,
    _dt: FT,
) {
    let num_particles = particles.position.len();
    let mut new_particle_id = num_particles;
    for i in 0..num_particles {
        if particles.particle_size_class[i] == ParticleSizeClass::TooLarge {
            let target_mass = particles.level_estimation[i].target_mass::<DU, D>(simulation_params);

            let mut num_children = FT::round(particles.mass[i] / target_mass) as usize;
            if num_children > split_patterns.get_max_num_children() {
                if simulation_params.fail_on_missing_split_pattern {
                    panic!("no split pattern for a 1-to-{} split", num_children);
                } else {
                    num_children = usize::min(num_children, split_patterns.get_max_num_children());
                }
            }
            assert!(num_children > 1);

            let split_pattern = split_patterns.get(num_children);

            let particle_radius = DU::sphere_volume_to_radius(particles.mass[i] / INIT_REST_DENSITY);

            let child_mass = particles.mass[i] / num_children as FT;
            let child_h_next = local_smoothing_length_from_mass::<DU, D>(child_mass, simulation_params.rest_density);
            let orig_velocity = particles.velocity[i];
            let orig_position = particles.position[i];
            let orig_level_estimation = particles.level_estimation[i];
            let orig_level_old = particles.level_old[i];

            let scale = particle_radius;

            particles.extend(num_children - 1);
            neighs.extend(num_children - 1);
            boundary_handler.extend(num_children - 1);

            for child_id in 0..num_children {
                if child_id == 0 {
                    particles.mass[i] = child_mass;
                    particles.velocity[i] = orig_velocity;
                    particles.position[i] = orig_position + split_pattern.pos_s[0] * scale;
                    particles.h2[i] = child_h_next;
                    particles.h2_next[i] = child_h_next;
                    particles.level_estimation[i] = orig_level_estimation;
                    particles.level_old[i] = orig_level_old;
                } else {
                    particles.mass[new_particle_id] = child_mass;
                    particles.velocity[new_particle_id] = orig_velocity;
                    particles.position[new_particle_id] = orig_position + split_pattern.pos_s[child_id] * scale;
                    particles.h2[i] = child_h_next;
                    particles.h2_next[new_particle_id] = child_h_next;
                    particles.level_estimation[new_particle_id] = orig_level_estimation;
                    particles.level_old[i] = orig_level_old;
                    new_particle_id += 1;
                }
            }
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct SplitPattern<const D: usize> {
    mass_s: Vec<FT>,
    pos_s: Vec<VF<D>>,
    h_s: Vec<FT>,
}

impl<const D: usize> SplitPattern<D> {
    pub fn assert_n_children(&self, n: usize) {
        assert_eq!(self.mass_s.len(), n);
        assert_eq!(self.pos_s.len(), n);
        assert_eq!(self.h_s.len(), n);
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct SplitPatterns<const D: usize>(Vec<SplitPattern<D>>);
impl<const D: usize> SplitPatterns<D> {
    pub fn new(v: Vec<SplitPattern<D>>) -> SplitPatterns<D> {
        for (i, sp) in v.iter().enumerate() {
            assert!(sp.pos_s.len() == i + 2);
        }
        println!("init with {} patterns", v.len() + 2);
        Self(v)
    }

    pub fn get(&self, num_children: usize) -> &SplitPattern<D> {
        assert!(num_children > 1);
        self.0
            .get(num_children - 2)
            .unwrap_or_else(|| panic!("no split pattern for a 1-to-{} split", num_children))
    }

    pub fn get_max_num_children(&self) -> usize {
        self.0.len() + 1
    }
}

enum InitialSplitPattern {
    RandomInSphere { radius: FT },
    #[allow(unused)]
    Spiral { factor: FT },
}

#[derive(PartialEq, Eq, Copy, Clone)]
enum SplitPatternState {
    Valid,
    RunawayParticle,
    Pairing,
}

/**
 * Suffixes/prefixes:
 *
 * o = original particle
 * s = split particles (s_count)
 * n = neighboring particles
 */
fn optimize_split_pattern<DU: DimensionUtils<D>, const D: usize>(
    pos_o: VF<D>,
    mass_o: FT,
    h_o: FT,
    s_count: usize,
    pos_n: &Vec<VF<D>>,
    mass_n: &Vec<FT>,
    h_n: &Vec<FT>,
    rest_density: FT,
    init_pattern: InitialSplitPattern,
    retries_count: usize,
    neighbors_distance: FT,
) -> (SplitPattern<D>, SplitPatternState) {
    let num_neighs = pos_n.len();
    assert!(mass_n.len() == num_neighs);
    assert!(h_n.len() == num_neighs);

    let child_mass = mass_o / s_count as FT;
    let child_h = local_smoothing_length_from_mass::<DU, D>(child_mass, INIT_REST_DENSITY);
    let mass_s = vec![child_mass; s_count];
    let h_s = vec![child_h as FT; s_count];

    let r = DU::sphere_volume_to_radius(mass_o / rest_density);

    let mut pos_s: Vec<VF<D>> = match D {
        2 => match init_pattern {
            InitialSplitPattern::RandomInSphere { radius } => {
                let mut rng = rand::thread_rng();

                (0..s_count)
                    .map(|_| {
                        let angle = rng.gen_range::<FT, _>(0. ..TAU);
                        let dist = rng.gen::<FT>().sqrt() * radius;

                        pos_o + VF::<D>::from_row_slice(&[FT::cos(angle), FT::sin(angle)]) * dist
                    })
                    .collect()
            }
            InitialSplitPattern::Spiral { factor } => {
                let angle = TAU / s_count as FT * factor;
                (0..s_count)
                    .map(|i| {
                        pos_o
                            + i as FT / s_count as FT
                                * 0.5
                                * r
                                * VF::<D>::from_row_slice(&[FT::cos(angle * i as FT), FT::sin(angle * i as FT)])
                    })
                    .collect()
            }
        },
        3 => {
            unimplemented!()
        }
        _ => {
            unreachable!()
        }
    };
    let mut new_pos_s = vec![VF::<D>::zeros(); s_count];

    let mut tau_s = vec![0.; s_count];
    let mut tau_n = vec![0.; num_neighs];

    let mut rho_o = mass_o * DU::kernelh(VF::<D>::zeros(), h_o);
    for n in 0..num_neighs {
        let xon = pos_o - pos_n[n];
        let hon = 0.5 * (h_o + h_n[n]);
        let won = DU::kernelh(xon, hon);
        rho_o += mass_n[n] * won;
    }

    assert!(rest_density * 0.999 < rho_o);
    assert!(rho_o < rest_density * 1.001);

    let mut iter_count = 0;
    loop {
        // evaluate tau_n
        for n in 0..num_neighs {
            let xno = pos_n[n] - pos_o;
            let hno = 0.5 * (h_n[n] + h_o);
            let wno = DU::kernelh(xno, hno);

            tau_n[n] = -mass_o * wno;
            for s in 0..s_count {
                let xns = pos_n[n] - pos_s[s];
                let hns = 0.5 * (h_n[n] + h_s[s]);
                let wns = DU::kernelh(xns, hns);
                tau_n[n] += mass_s[s] * wns;
            }
        }

        // evaulate tau_s
        for s in 0..s_count {
            tau_s[s] = -rho_o;
            for k in 0..s_count {
                let xsk = pos_s[s] - pos_s[k];
                let hsk = 0.5 * (h_s[s] + h_s[k]);
                let wsk = DU::kernelh(xsk, hsk);
                tau_s[s] += mass_s[k] * wsk;
            }
            for n in 0..num_neighs {
                let xsn = pos_s[s] - pos_n[n];
                let hsn = 0.5 * (h_s[s] + h_n[n]);
                let wsn = DU::kernelh(xsn, hsn);
                tau_s[s] += mass_n[n] * wsn;
            }
        }

        // evaulate error
        if false {
            let en: FT = (0..num_neighs).map(|n| mass_n[n] * tau_n[n] * tau_n[n]).sum();
            let es: FT = (0..s_count).map(|s| mass_s[s] * tau_s[s] * tau_s[s]).sum();

            println!("pattern {} - iter count {}: en={} es={}", s_count, iter_count, en, es);
        }

        /*
        if false {
            let mut data_ft: Vec<(String, Vec<FT>)> = Vec::new();
            let data_vec: Vec<(String, Vec<VF<D>>)> = Vec::new();
            let mut data_u8: Vec<(String, Vec<u8>)> = Vec::new();
            let lines = Vec::new();

            data_ft.push((
                "mass".into(),
                mass_s.iter().cloned().chain(mass_n.iter().cloned()).collect(),
            ));
            data_ft.push((
                "tau".into(),
                tau_s.iter().cloned().chain(tau_n.iter().cloned()).collect(),
            ));
            data_u8.push((
                "is_child".into(),
                repeat(1).take(s_count).chain(repeat(0).take(num_neighs)).collect(),
            ));

            create_dir_all(format!("./split-patterns/{:03}-{:02}", s_count, retries_count)).unwrap();

            write_vtk_file2(
                format!(
                    "./split-patterns/{:03}-{:02}/pattern-{}.vtk",
                    s_count, retries_count, iter_count
                ),
                pos_s
                    .iter()
                    .cloned()
                    .chain(pos_n.iter().cloned())
                    .collect::<Vec<VF<D>>>(),
                data_ft,
                data_vec,
                data_u8,
                lines,
            );
        }
        */

        // evaluate new positions for all split particles i
        for i in 0..s_count {
            let mut pos_grad = VF::<D>::zeros();

            // Optimized Refinement for Spatially Adaptive SPH Eq 12
            for n in 0..num_neighs {
                let xin = pos_s[i] - pos_n[n];
                let hin = 0.5 * (h_s[i] + h_n[n]);
                let win = DU::kernel_derivh(xin, hin);
                pos_grad += mass_n[n] * (tau_s[i] + tau_n[n]) * win;
            }

            // Optimized Refinement for Spatially Adaptive SPH Eq 12
            for s in 0..s_count {
                let xis = pos_s[i] - pos_s[s];
                let his = 0.5 * (h_s[i] + h_s[s]);
                let wis = DU::kernel_derivh(xis, his);
                pos_grad += mass_s[s] * (tau_s[i] + tau_s[s]) * wis;
            }

            pos_grad *= 2. * mass_s[i];

            let step_size = 0.01;
            let displacement = -pos_grad * step_size;
            // if iter_count == 10000 {
            //     println!("{}@iter {}: displacement {} step size {}", s_count, iter_count, displacement.norm(), step_size);
            // }
            // let max_displacement = 0.09 * r;
            // if displacement.norm_squared() > max_displacement * max_displacement {
            //     displacement *= max_displacement / (displacement.norm() + 0.0001);
            // }

            new_pos_s[i] = pos_s[i] + displacement;
        }

        swap(&mut pos_s, &mut new_pos_s);

        // validate that no two particles are too close to each other
        if iter_count > 1000 && iter_count % 200 == 0 {
            let min_req_dist = 0.1 * DU::sphere_volume_to_radius(child_mass / rest_density);
            for n1 in 0..pos_s.len() {
                for n2 in (n1 + 1)..pos_s.len() {
                    let distance = pos_s[n1] - pos_s[n2];
                    if distance.norm_squared() < min_req_dist {
                        // restart simulation with different initial pattern
                        println!("child:{} iter:{} restart due particle merging", s_count, iter_count);
                        return (SplitPattern::<D> { mass_s, h_s, pos_s }, SplitPatternState::Pairing);
                    }
                }
            }

            let max_displacement = pos_s
                .iter()
                .map(|x| x.norm_squared())
                .max_by(|x, y| x.partial_cmp(&y).unwrap())
                .unwrap()
                .sqrt();

            // validate that no child particle has left the "original area" of the parent particle of the parent particle of the parent particle of the parent particle
            if max_displacement > neighbors_distance * 0.99 {
                // restart simulation with different initial pattern
                println!(
                    "child:{} retry:{} iter:{} particle out of valid region -> restart optimization",
                    s_count, retries_count, iter_count
                );
                return (
                    SplitPattern::<D> { mass_s, h_s, pos_s },
                    SplitPatternState::RunawayParticle,
                );
            }

            if iter_count >= 40000 {
                return (SplitPattern::<D> { mass_s, h_s, pos_s }, SplitPatternState::Valid);
            }
        }

        iter_count += 1;
    }
}

/**
 * Generate point set where a point has the coordinates "zero" and
 * the space is packed by the vertices of a covering of triangles(2D)/tetrahedra(3D)
 */
fn generate_tetrahedral_point_set<const D: usize>(distance: FT, min: VF<D>, max: VF<D>) -> Vec<VF<D>> {
    match D {
        2 => {
            let mut points: Vec<VF<D>> = Vec::new();
            let h = FT::sqrt(3.) * 0.5;

            let rmin = (min[1] / h).ceil() as i32;
            let rmax = (max[1] / h).floor() as i32;

            for row in rmin..=rmax {
                let y = h * row as FT;
                let cshift = if row % 2 == 0 { 0. } else { distance / 2. };

                let cmin = ((min[0] - cshift) / distance).ceil() as i32;
                let cmax = ((max[0] - cshift) / distance).floor() as i32;

                for column in cmin..=cmax {
                    points.push(VF::<D>::from_row_slice(&[cshift + column as FT * distance, y]));
                }
            }

            points
        }
        3 => {
            unimplemented!()
        }
        _ => {
            unreachable!()
        }
    }
}

fn find_optimal_mass<DU: DimensionUtils<D>, const D: usize>(
    initial_mass: FT,
    rest_density: FT,
    positions: &[VF<D>],
) -> FT {
    let mut mass = initial_mass;
    let mut mass_update = initial_mass;
    let max_distance = positions
        .iter()
        .map(|x| x.norm())
        .max_by(|x, y| x.partial_cmp(&y).unwrap())
        .unwrap();
    for _ in 0..40 {
        // calculate density for grid setting with given mass
        let h = local_smoothing_length_from_mass::<DU, D>(mass, rest_density);

        // assure that enough neighbors are supplied for neighbor calculations
        assert!(h < max_distance);

        let mut density = 0.;
        for &x in positions {
            density += mass * DU::kernelh(x, h);
        }

        // println!(
        //     "iter {}: density={} rest={} mass={}",
        //     num_iter, density, rest_density, mass
        // );
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

pub fn precalculate_split_pattern<DU: DimensionUtils<D>, const D: usize>(
    num_children: usize,
    rest_density: FT,
    export_svg: bool
) -> SplitPattern<D> {
    println!();
    println!("===============================");
    println!("== {} children =============", num_children);

    let bound_ft = 2.
        * DU::support_radius_by_smoothing_length()
        * local_smoothing_length_from_volume::<DU, D>(DU::radius_to_sphere_volume(1.));

    let min = VF::<D>::from_iterator(repeat(-bound_ft));
    let max = VF::<D>::from_iterator(repeat(bound_ft));

    let mut neighbors_distance = 1.;
    let mut pos_n = generate_tetrahedral_point_set(neighbors_distance, min, max);

    let mut mass = find_optimal_mass::<DU, D>(1., rest_density, &pos_n);

    // ----------------------------------------------------------------------------------------------------
    // scale everything so that the particle radius is 1
    let r = DU::sphere_volume_to_radius(mass / rest_density);
    for p in &mut pos_n {
        *p *= 1. / r;
    }
    neighbors_distance *= 1. / r;
    // println!("{}", DU::radius_to_sphere_volume(1.) * rest_density);
    // panic!("{}", mass * DU::radius_to_sphere_volume(1. / r));
    mass = DU::radius_to_sphere_volume(1.) * rest_density;
    // ----------------------------------------------------------------------------------------------------

    let h = local_smoothing_length_from_mass::<DU, D>(mass, rest_density);

    let (origin_pos_id, origin_pos_norm) = pos_n
        .iter()
        .enumerate()
        .map(|(i, x)| (i, x.norm_squared()))
        .min_by(|(_, x), (_, y)| x.partial_cmp(&y).unwrap())
        .unwrap();
    assert!(origin_pos_norm == 0.);

    // drop neighbor at origin
    let pos_o = pos_n.remove(origin_pos_id);

    let mass_n = vec![mass; pos_n.len()];
    let h_n = vec![h; pos_n.len()];

    for iter in 0..300 {
        let (split_pattern, split_pattern_state) = optimize_split_pattern::<DU, D>(
            pos_o,
            mass,
            h,
            num_children,
            &pos_n,
            &mass_n,
            &h_n,
            rest_density,
            // InitialSplitPattern::Spiral {
            //     factor: (iter + 1) as FT,
            // },
            InitialSplitPattern::RandomInSphere { radius: 0.6 },
            iter,
            neighbors_distance,
        );

        if export_svg {
            draw_svg::<DU, D>(
                &pos_n,
                mass,
                rest_density,
                &split_pattern,
                split_pattern_state,
                num_children,
                iter,
            );
        }

        if split_pattern_state == SplitPatternState::Valid {
            return split_pattern;
        }
    }

    panic!("no valid split pattern found num_children={}", num_children);
}

fn draw_svg<DU: DimensionUtils<D>, const D: usize>(
    pos_n: &Vec<VF<D>>,
    mass: FT,
    rest_density: FT,
    split_pattern: &SplitPattern<D>,
    split_pattern_state: SplitPatternState,
    num_children: usize,
    iter: usize,
) {
    {
        let svgscale = 50.;

        let minx = pos_n
            .iter()
            .map(|p| p[0])
            .min_by(|x, y| x.partial_cmp(&y).unwrap())
            .unwrap();
        let miny = pos_n
            .iter()
            .map(|p| p[1])
            .min_by(|x, y| x.partial_cmp(&y).unwrap())
            .unwrap();
        let maxx = pos_n
            .iter()
            .map(|p| p[0])
            .max_by(|x, y| x.partial_cmp(&y).unwrap())
            .unwrap();
        let maxy = pos_n
            .iter()
            .map(|p| p[1])
            .max_by(|x, y| x.partial_cmp(&y).unwrap())
            .unwrap();

        let viewbox_fraction = 0.2;

        let mut document = Document::new().set(
            "viewBox",
            format!(
                "{} {} {} {}",
                minx * svgscale * viewbox_fraction,
                miny * svgscale * viewbox_fraction,
                (maxx - minx) * svgscale * viewbox_fraction,
                (maxy - miny) * svgscale * viewbox_fraction
            ),
        );

        for x in pos_n {
            let r = DU::sphere_volume_to_radius(mass / rest_density);
            let stroke_width = r * 0.1;
            document = document.add(
                Circle::new()
                    .set("fill", "gray")
                    .set("stroke", "black")
                    .set("stroke-width", stroke_width * svgscale)
                    .set("cx", x[0] * svgscale)
                    .set("cy", x[1] * svgscale)
                    .set("r", (r - stroke_width * 0.5) * svgscale),
            );
        }

        for i in 0..split_pattern.pos_s.len() {
            let x = split_pattern.pos_s[i];
            let mass = split_pattern.mass_s[i];
            let r = DU::sphere_volume_to_radius(mass / rest_density);
            let stroke_width = r * 0.25;
            document = document.add(
                Circle::new()
                    .set("fill", "blue")
                    .set("stroke", "black")
                    .set("stroke-width", stroke_width * svgscale)
                    .set("cx", x[0] * svgscale)
                    .set("cy", x[1] * svgscale)
                    .set("r", (r - stroke_width * 0.5) * svgscale),
            );
        }

        create_dir_all(format!("./split-patterns")).unwrap();

        let state_str = match split_pattern_state {
            SplitPatternState::Pairing => "paired",
            SplitPatternState::Valid => "valid",
            SplitPatternState::RunawayParticle => "runaway",
        };

        svg::save(
            format!("./split-patterns/{:03}-{:02}-{}.svg", num_children, iter, state_str),
            &document,
        )
        .unwrap();
    }
}
