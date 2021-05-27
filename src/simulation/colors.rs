use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
};

use crate::{
    adaptivity::ParticleSizeClass,
    color_map::ColorMap,
    floating_type_mod::FT,
    neighborhood_search::NeighborhoodCache,
    optimal_neighbor_number,
    simulation_parameters::SimulationParams,
    sph_kernels::{smoothing_length_single, DimensionUtils},
    vec3f, LevelEstimationState, ParticleVec, VisualizationParams, VisualizedAttribute, V,
};

pub fn color_map_inferno(min: FT, max: FT) -> ColorMap {
    ColorMap::new(vec![
        (
            min + (max - min) * 0.0,
            vec3f(0.0014619955811715805, 0.0004659913919114934, 0.013866005775115809),
        ),
        (
            min + (max - min) * 0.06666666666666667,
            vec3f(0.04691458399133113, 0.030323540520811973, 0.15016326468244964),
        ),
        (
            min + (max - min) * 0.13333333333333333,
            vec3f(0.14237847430795506, 0.04624117675574093, 0.30855378680836465),
        ),
        (
            min + (max - min) * 0.2,
            vec3f(0.2582339375612672, 0.038569281262784215, 0.4064850812186898),
        ),
        (
            min + (max - min) * 0.26666666666666666,
            vec3f(0.366528457743285, 0.07157684449494817, 0.4319940445656597),
        ),
        (
            min + (max - min) * 0.3333333333333333,
            vec3f(0.47232856222023284, 0.11054509253877559, 0.428334014815688),
        ),
        (
            min + (max - min) * 0.4,
            vec3f(0.5783040710826255, 0.1480366969821801, 0.4044110859921461),
        ),
        (
            min + (max - min) * 0.4666666666666667,
            vec3f(0.6826555952415246, 0.1894982847225483, 0.3607573457624624),
        ),
        (
            min + (max - min) * 0.5333333333333333,
            vec3f(0.780517595641067, 0.24332476411029125, 0.29952273568573573),
        ),
        (
            min + (max - min) * 0.6,
            vec3f(0.865006157141607, 0.316819514079576, 0.2260550749407627),
        ),
        (
            min + (max - min) * 0.6666666666666666,
            vec3f(0.9296439014941755, 0.41147612778815296, 0.14536750158970949),
        ),
        (
            min + (max - min) * 0.7333333333333333,
            vec3f(0.970919318954511, 0.5228513513717987, 0.05836666742473027),
        ),
        (
            min + (max - min) * 0.8,
            vec3f(0.987622172670732, 0.6453178289458518, 0.039886017500422775),
        ),
        (
            min + (max - min) * 0.8666666666666667,
            vec3f(0.9788062634501479, 0.7745421938654863, 0.1760361942373471),
        ),
        (
            min + (max - min) * 0.9333333333333333,
            vec3f(0.950018012245954, 0.9034074125145412, 0.3802723264284489),
        ),
        (
            min + (max - min) * 1.0,
            vec3f(0.9883620799212208, 0.9983616470620554, 0.6449240982803861),
        ),
    ])
}

pub fn color_map_viridis(min: FT, max: FT) -> ColorMap {
    ColorMap::new(vec![
        (
            min + (max - min) * 0.0,
            vec3f(0.2670039853213788, 0.0048725657145795975, 0.32941506855247793),
        ),
        (
            min + (max - min) * 0.06666666666666667,
            vec3f(0.28265591676374746, 0.10019440706631136, 0.42215967285462885),
        ),
        (
            min + (max - min) * 0.13333333333333333,
            vec3f(0.27713381181214125, 0.18522747944269774, 0.4898983578428951),
        ),
        (
            min + (max - min) * 0.2,
            vec3f(0.25393482507335086, 0.26525311670734747, 0.529983099667603),
        ),
        (
            min + (max - min) * 0.26666666666666666,
            vec3f(0.22198891605799553, 0.33915975136273824, 0.5487520417750932),
        ),
        (
            min + (max - min) * 0.3333333333333333,
            vec3f(0.19063051802725675, 0.4070603881536437, 0.5560891205440711),
        ),
        (
            min + (max - min) * 0.4,
            vec3f(0.1636245598287687, 0.47113199888460483, 0.5581480982786068),
        ),
        (
            min + (max - min) * 0.4666666666666667,
            vec3f(0.13914656229528236, 0.5338106140906136, 0.555298125858835),
        ),
        (
            min + (max - min) * 0.5333333333333333,
            vec3f(0.12056429075653713, 0.5964211612480832, 0.5436109978665574),
        ),
        (
            min + (max - min) * 0.6,
            vec3f(0.1346914034616326, 0.6586347623899736, 0.5176490803131216),
        ),
        (
            min + (max - min) * 0.6666666666666666,
            vec3f(0.2080291781284243, 0.7186993731402823, 0.47287333975819085),
        ),
        (
            min + (max - min) * 0.7333333333333333,
            vec3f(0.32779655496333804, 0.7739788075712202, 0.40663965647349865),
        ),
        (
            min + (max - min) * 0.8,
            vec3f(0.47750397699915853, 0.8214424087022711, 0.3181950138984179),
        ),
        (
            min + (max - min) * 0.8666666666666667,
            vec3f(0.6472561782044223, 0.8583980753432965, 0.2098615478515251),
        ),
        (
            min + (max - min) * 0.9333333333333333,
            vec3f(0.8249409891695173, 0.8847181273467387, 0.10621658195896774),
        ),
        (
            min + (max - min) * 1.0,
            vec3f(0.9932481489335602, 0.9061547634208059, 0.14393594366968385),
        ),
    ])
}

// https://www.kennethmoreland.com/color-advice/
pub fn color_map_smooth_warm_cool(min: FT, max: FT) -> ColorMap {
    ColorMap::new(vec![
        (
            min + (max - min) * 0.0,
            vec3f(0.22999950386952345, 0.2989989340493756, 0.754000138575591),
        ),
        (
            min + (max - min) * 0.06666666666666667,
            vec3f(0.3092628286343541, 0.41381741502624314, 0.8506465305229377),
        ),
        (
            min + (max - min) * 0.13333333333333333,
            vec3f(0.39425737212145806, 0.5228162411749818, 0.9256191958236355),
        ),
        (
            min + (max - min) * 0.2,
            vec3f(0.48411429029138214, 0.6225453939820256, 0.9756171248512094),
        ),
        (
            min + (max - min) * 0.26666666666666666,
            vec3f(0.5763688847791901, 0.7093314701279408, 0.9985547366074979),
        ),
        (
            min + (max - min) * 0.3333333333333333,
            vec3f(0.6676027122061123, 0.7797067894635203, 0.9936255761414589),
        ),
        (
            min + (max - min) * 0.4,
            vec3f(0.7539223294202332, 0.8306352658223072, 0.9612951081820061),
        ),
        (
            min + (max - min) * 0.4666666666666667,
            vec3f(0.8313102065328385, 0.8596668872539794, 0.9032226182187731),
        ),
        (
            min + (max - min) * 0.5333333333333333,
            vec3f(0.8997903661548088, 0.8476560828846528, 0.8178272445677435),
        ),
        (
            min + (max - min) * 0.6,
            vec3f(0.9479258586405416, 0.7951017243636899, 0.717097358491085),
        ),
        (
            min + (max - min) * 0.6666666666666666,
            vec3f(0.9689983985927076, 0.7213814889843815, 0.612361864520328),
        ),
        (
            min + (max - min) * 0.7333333333333333,
            vec3f(0.9635890818164178, 0.6287902448925465, 0.507610242858806),
        ),
        (
            min + (max - min) * 0.8,
            vec3f(0.9326959127654463, 0.5196409344481697, 0.40634608316087356),
        ),
        (
            min + (max - min) * 0.8666666666666667,
            vec3f(0.8778897629242048, 0.3952016942845104, 0.3115039838993836),
        ),
        (
            min + (max - min) * 0.9333333333333333,
            vec3f(0.8013768295142913, 0.25153661146923556, 0.2254356213476219),
        ),
        (
            min + (max - min) * 1.0,
            vec3f(0.7060001359117047, 0.015991824033980695, 0.15000007192220008),
        ),
    ])
}

// https://www.kennethmoreland.com/color-advice/
pub fn black_body_color_map(min: FT, max: FT) -> ColorMap {
    ColorMap::new(vec![
        (min + (max - min) * 0.0, vec3f(0.0, 0.0, 0.0)),
        (
            min + (max - min) * 0.06666666666666667,
            vec3f(0.1394282635051205, 0.06009480851170646, 0.035776007200922486),
        ),
        (
            min + (max - min) * 0.13333333333333333,
            vec3f(0.2413944229177235, 0.08558924491193284, 0.06580307931909093),
        ),
        (
            min + (max - min) * 0.2,
            vec3f(0.35192321947598326, 0.10480903601879557, 0.08480286230027789),
        ),
        (
            min + (max - min) * 0.26666666666666666,
            vec3f(0.4681633134778857, 0.11943965573140711, 0.1017091011689388),
        ),
        (
            min + (max - min) * 0.3333333333333333,
            vec3f(0.5893191205062877, 0.1292325944480741, 0.11862742556708483),
        ),
        (
            min + (max - min) * 0.4,
            vec3f(0.706849160518104, 0.15003367886528432, 0.13144472110159658),
        ),
        (
            min + (max - min) * 0.4666666666666667,
            vec3f(0.7732082716359728, 0.25503160482399895, 0.11167779691868823),
        ),
        (
            min + (max - min) * 0.5333333333333333,
            vec3f(0.8393857201928158, 0.3455445563935293, 0.07610402711123557),
        ),
        (
            min + (max - min) * 0.6,
            vec3f(0.8937766259341077, 0.4398522418561916, 0.027932126084511257),
        ),
        (
            min + (max - min) * 0.6666666666666666,
            vec3f(0.9059845490140521, 0.5542965728309169, 0.07074279841591524),
        ),
        (
            min + (max - min) * 0.7333333333333333,
            vec3f(0.91246942542167, 0.6620924424620748, 0.11226006866305718),
        ),
        (
            min + (max - min) * 0.8,
            vec3f(0.9126417956482176, 0.7666648306494336, 0.15338563119383095),
        ),
        (
            min + (max - min) * 0.8666666666666667,
            vec3f(0.9057490741507888, 0.8696584198795938, 0.19472348449042068),
        ),
        (
            min + (max - min) * 0.9333333333333333,
            vec3f(0.9632876594878218, 0.9402577467240134, 0.5654314125128757),
        ),
        (min + (max - min) * 1.0, vec3f(1.0, 1.0, 1.0)),
    ])
}

pub fn get_color_map_for_pressure<DU: DimensionUtils<D>, const D: usize>(
    _attr: VisualizedAttribute,
    _simulation_params: SimulationParams,
    max_pressure: FT,
) -> ColorMap {
    ColorMap::new(vec![
        (0., vec3f(1., 1., 1.)), //
        (max_pressure, vec3f(1., 0., 0.)),
    ])
}

pub fn get_color_map<DU: DimensionUtils<D>, const D: usize>(
    attr: VisualizedAttribute,
    simulation_params: SimulationParams,
) -> Option<ColorMap> {
    match attr {
        VisualizedAttribute::SourceTerm => Some(color_map_viridis(-6000., 6000.)),
        VisualizedAttribute::Aii => Some(ColorMap::new(vec![
            (-1.0, vec3f(1., 0., 0.)),
            (0.0, vec3f(1., 1., 1.)),
            (50.0, vec3f(0., 0., 1.)),
        ])),
        VisualizedAttribute::Distance => {
            Some(color_map_inferno(-simulation_params.maximum_surface_distance, 0.))
            // Some(ColorMap::new(vec![
            //     (-1.1, vec3f(1., 0., 1.)),
            //     (-1.0, vec3f(0., 1., 1.)),
            //     (-0.9, vec3f(1., 1., 0.)),
            //     (-0.8, vec3f(0., 0., 1.)),
            //     (-0.7, vec3f(1., 0., 0.)),
            //     (-0.6, vec3f(1., 0., 1.)),
            //     (-0.5, vec3f(0., 1., 1.)),
            //     (-0.4, vec3f(1., 1., 0.)),
            //     (-0.3, vec3f(0., 0., 1.)),
            //     (-0.2, vec3f(1., 0., 0.)),
            //     (-0.1, vec3f(0., 1., 0.)),
            //     (-0.0, vec3f(0., 0., 1.)), //
            // ]))
        }
        VisualizedAttribute::Velocity => {
            // Some(ColorMap::new(vec![
            //     (0., vec3f(0., 0., 1.)), //
            //     (1., vec3f(1., 1., 1.)),
            // ]))

            // Some(black_body_color_map(0., 4.))
            Some(color_map_viridis(0., 4.))
        }
        VisualizedAttribute::Density => {
            Some(ColorMap::new(vec![
                (0.9, vec3f(0., 0., 1.)), //
                (1., vec3f(1., 1., 1.)),
                (1.01, vec3f(1., 0., 0.)),
            ]))
        }
        VisualizedAttribute::NeighborCount => {
            Some(ColorMap::new(vec![
                (-4., vec3f(0., 0., 1.)), // blue
                (-2., vec3f(0., 1., 1.)), // turkoise
                // (-0.5, vec3f(0., 1., 0.)), // white
                (0., vec3f(0., 1., 0.)), // white
                // (0.5, vec3f(0., 1., 0.)), // white
                (2., vec3f(1., 1., 0.)), // yellow
                (4., vec3f(1., 0., 0.)), // red
            ]))
            // Some(color_map_smooth_warm_cool(-4., 4.))
        }
        VisualizedAttribute::ConstantField => {
            let diff = 1.05;
            Some(ColorMap::new(vec![
                (2. - diff, vec3f(0., 0., 1.)), //
                (1., vec3f(1., 1., 1.)),
                (diff, vec3f(1., 0., 0.)),
            ]))
        }
        VisualizedAttribute::MinDistanceToNeighbor => {
            Some(ColorMap::new(vec![
                (0.0, vec3f(1., 0., 0.)), //
                (0.1, vec3f(1., 1., 0.)), //
                (0.3, vec3f(0., 1., 0.)), //
                (1.0, vec3f(0., 0., 1.)),
                (1.2, vec3f(1., 0., 1.)),
            ]))
        }
        VisualizedAttribute::Pressure
        | VisualizedAttribute::RandomColor
        | VisualizedAttribute::SingleColor
        | VisualizedAttribute::ParticleSizeClass => None,
    }
}

pub fn get_color_for_particle<DU: DimensionUtils<D>, const D: usize>(
    i: usize,
    particles: &ParticleVec<D>,
    simulation_params: SimulationParams,
    max_pressure: Option<FT>,
    neighs: &NeighborhoodCache,
    visualization_params: VisualizationParams,
) -> V<f64, 3> {
    if particles.flag_neighborhood_reduced[i] && visualization_params.show_flag_neighborhood_reduced {
        return [0., 1., 0.].into();
    }

    if particles.flag_is_fluid_surface[i] && visualization_params.show_flag_is_fluid_surface {
        return [1., 0., 0.].into();
    }

    if particles.flag_insufficient_neighs[i] && visualization_params.show_flag_is_fluid_surface {
        return [0., 1., 0.].into();
    }

    let attr = visualization_params.visualized_attribute;
    match attr {
        VisualizedAttribute::Aii => {
            let color_map = get_color_map::<DU, D>(attr, simulation_params).unwrap();
            color_map.get_f64(particles.aii[i])
        }
        VisualizedAttribute::Distance => {
            let color_map = get_color_map::<DU, D>(attr, simulation_params).unwrap();
            let dist;
            if !visualization_params.take_data_from_stash {
                dist = match particles.level_estimation[i] {
                    LevelEstimationState::FluidInterior => -simulation_params.maximum_surface_distance,
                    LevelEstimationState::FluidSurface(dist) => dist,
                };
            } else {
                dist = particles.stash[i];
            }

            color_map.get_f64(dist)
        }
        VisualizedAttribute::Pressure => {
            let color_map = get_color_map_for_pressure::<DU, D>(attr, simulation_params, max_pressure.unwrap());

            color_map.get_f64(particles.pressure[i])
        }
        VisualizedAttribute::Velocity => {
            let color_map = get_color_map::<DU, D>(attr, simulation_params).unwrap();

            color_map.get_f64(particles.velocity[i].norm())
        }
        VisualizedAttribute::Density => {
            let color_map = get_color_map::<DU, D>(attr, simulation_params).unwrap();
            color_map.get_f64(particles.density[i] / simulation_params.rest_density)
        }
        VisualizedAttribute::NeighborCount => {
            let count = particles.neighbor_count[i];
            let baseline = optimal_neighbor_number::<DU, D>();
            // println!("{}", count);

            let color_map = get_color_map::<DU, D>(attr, simulation_params).unwrap();

            color_map.get_f64(count as FT - baseline)
        }
        VisualizedAttribute::RandomColor => {
            let mut s = DefaultHasher::new();
            i.hash(&mut s);
            let v: u64 = s.finish();

            let r = ((v >> 0) & 0xFF) as u8;
            let g = ((v >> 8) & 0xFF) as u8;
            let b = ((v >> 16) & 0xFF) as u8;

            [r as f64 / 255., g as f64 / 255., b as f64 / 255.].into()
        }
        VisualizedAttribute::ConstantField => {
            let color_map = get_color_map::<DU, D>(attr, simulation_params).unwrap();
            color_map.get_f64(particles.constant_field[i])
        }
        VisualizedAttribute::MinDistanceToNeighbor => {
            let v = neighs
                .iter(i)
                .filter(|&j| j != i)
                .map(|j| {
                    let dist = (particles.position[i] - particles.position[j]).norm();
                    dist / smoothing_length_single(&particles.h2, i, simulation_params)
                })
                .chain(std::iter::once(2.))
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();

            let color_map = get_color_map::<DU, D>(attr, simulation_params).unwrap();

            color_map.get_f64(v)
        }
        VisualizedAttribute::ParticleSizeClass => match particles.particle_size_class[i] {
            // blue
            ParticleSizeClass::TooSmall => [0., 0., 1.].into(),
            // light blue
            ParticleSizeClass::Small => [0.5, 0.5, 1.].into(),
            // white
            ParticleSizeClass::Optimal => [1., 1., 1.].into(),
            // light red
            ParticleSizeClass::Large => [1., 0.5, 0.5].into(),
            // red
            ParticleSizeClass::TooLarge => [1., 0., 0.].into(),
        },
        VisualizedAttribute::SingleColor => [80. / 255., 140. / 255., 255. / 255.].into(),
        VisualizedAttribute::SourceTerm => {
            let color_map = get_color_map::<DU, D>(attr, simulation_params).unwrap();
            color_map.get_f64(particles.ppe_source_term[i])
        }
    }
}
