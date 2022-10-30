use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    path::Path,
    sync::{Arc, Mutex},
    time::Duration,
};

use clap::{App, AppSettings, Arg, SubCommand};
use eframe::epaint::ahash::HashMap;

use crate::{
    boundary_handler::BoundaryHandler, floating_type_mod::FT, generate_split_patterns, init_fluid_sim,
    init_simulation_params, load_split_patterns_from_file, properties_window::properties_window_main,
    simulation_parameters::SimulationParams, sph_kernels::DimensionUtils2d, write_split_patterns_to_file,
    write_statistics, FluidSimulation, SceneConfig, SimulationVisualizer, VisualizationParams,
};

use super::{animation, rendering::SimulationWindow, video_encoder::VideoEncoder, vtk_exporter::VtkExporter};

const CARGO_PKG_AUTHORS: &'static str = env!("CARGO_PKG_AUTHORS");
const CARGO_PKG_VERSION: &'static str = env!("CARGO_PKG_VERSION");
const CARGO_PKG_DESCRIPTION: &'static str = env!("CARGO_PKG_DESCRIPTION");

pub fn start() {
    let matches = App::new("Adaptive SPH Simulation")
        .version(CARGO_PKG_VERSION)
        .author(CARGO_PKG_AUTHORS)
        .about(CARGO_PKG_DESCRIPTION)
        .setting(AppSettings::SubcommandRequiredElseHelp)
        .arg(
            Arg::with_name("v")
                .short("v")
                .multiple(true)
                .help("Sets the level of verbosity"),
        )
        .subcommand(
            SubCommand::with_name("run")
                .about("Run simulation with given config")
                .arg(
                    Arg::with_name("SIMULATION_CONFIG")
                        .help("Sets the simulation paramaters")
                        .required(true)
                        .index(1),
                )
                .arg(
                    Arg::with_name("SCENE_CONFIG")
                        .help("Scene setup")
                        .required(true)
                        .index(2),
                )
                .arg(
                    Arg::with_name("MAX_SECONDS")
                        .long("max-seconds")
                        .short("s")
                        .required(false)
                        .takes_value(true)
                        .help("Stop simulation after the given amount of seconds"),
                )
                .arg(
                    Arg::with_name("OVERWRITE_CONFIG_FILE")
                        .long("overwrite-config-file")
                        .short("c")
                        .required(false)
                        .takes_value(true)
                        .help("Overwrite config"),
                )
                .arg(
                    Arg::with_name("STATISTICS_ENABLED")
                        .help("Track performance of individual steps")
                        .short("p")
                        .long("statistics-enabled")
                        .takes_value(false)
                )
                .arg(
                    Arg::with_name("STATISTICS_PATH")
                        .long("statistics-path")
                        .short("w")
                        .required(false)
                        .takes_value(true)
                        .help("Where to write statistics to"),
                ),
        )
        .subcommand(
            SubCommand::with_name("image")
                .about("Run simulation with given config")
                .arg(
                    Arg::with_name("IMAGE_EXPORT_CONFIG_FILES")
                        .help("Sets the simulation paramaters")
                        .multiple(true),
                ),
        )
        .subcommand(
            SubCommand::with_name("generate-split-patterns")
                .about("Generate split patterns (takes a while) and save result to file. Split patterns are needed prior to running the fluid simulation.")
                .arg(
                    Arg::with_name("OUTPUT_YAML")
                        .help("YAML file where the split patterns are written to")
                        .default_value("./split-patterns.yaml")
                        .takes_value(true)
                        .required(true)
                ),
        )
        .get_matches();

    if let Some(run_matches) = matches.subcommand_matches("run") {
        let parameter_file = run_matches
            .value_of("SIMULATION_CONFIG")
            .expect("missing simulation config");
        let params_yaml = std::fs::read_to_string(parameter_file).expect("failed reading parameter file");
        let mut simulation_params_serde: serde_yaml::Value =
            serde_yaml::from_str(&params_yaml).expect("failed parsing simulation config file");

        if let Some(overwrite_value_config) = run_matches.value_of("OVERWRITE_CONFIG_FILE") {
            let overwrite_config_str =
                std::fs::read_to_string(overwrite_value_config).expect("failed reading parameter file");
            let overwrite_config_file: HashMap<String, serde_yaml::Value> =
                serde_yaml::from_str(&overwrite_config_str).expect("failed parsing simulation config file");
            for (k, v) in overwrite_config_file.into_iter() {
                let mapping = simulation_params_serde
                    .as_mapping_mut()
                    .expect("cannot get parsed simulation parameters as mapping");
                *mapping
                    .get_mut(&serde_yaml::Value::String(k.clone()))
                    .unwrap_or_else(|| panic!("not able to find attribute {}", k)) = v;
            }
        }

        let simulation_params: SimulationParams =
            serde_yaml::from_value(simulation_params_serde).expect("failed to unpack SimulationParams");
        println!("{:?}", simulation_params);

        let scene_file_path = run_matches.value_of("SCENE_CONFIG").expect("missing scene config");
        let scene_yaml = std::fs::read_to_string(scene_file_path).expect("failed reading scene file");
        let scene_config: SceneConfig = serde_yaml::from_str(&scene_yaml).expect("failed parsing scene config file");
        println!("{:?}", scene_config);

        let cancel: Arc<Mutex<bool>> = Arc::new(Mutex::from(false));
        let restart: Arc<Mutex<bool>> = Arc::new(Mutex::from(false));

        // panic!("{}", <DimensionUtils2d as DimensionUtils<2>>::sphere_volume_to_radius(<DimensionUtils2d as DimensionUtils<2>>::optimal_neighbor_number() as FT * INIT_PARTICLE_VOLUME)  / simulation_params.kernel_support_radius);

        let properties: Arc<Mutex<SimulationParams>> = Arc::new(Mutex::from(simulation_params));

        let visualization_params: VisualizationParams = VisualizationParams::default();
        let visualization_params_arc: Arc<Mutex<VisualizationParams>> = Arc::new(Mutex::from(visualization_params));

        let counters_enabled = run_matches.is_present("STATISTICS_ENABLED");

        let max_seconds = run_matches.value_of("MAX_SECONDS").map(|x| x.parse::<FT>().unwrap());
        let statistics_path_opt = run_matches.value_of("STATISTICS_PATH").map(String::from);

        let fluid_thread = {
            let cancel = cancel.clone();
            let restart = restart.clone();
            let properties = properties.clone();
            let visualization_params_arc = visualization_params_arc.clone();
            std::thread::spawn(move || {
                let _result = std::panic::catch_unwind(|| {
                    let fluid_simulation = fluid_main(
                        cancel.clone(),
                        restart,
                        properties,
                        &scene_config,
                        visualization_params_arc,
                        max_seconds,
                        counters_enabled,
                    )
                    .unwrap();

                    if counters_enabled {
                        let s = write_statistics(&fluid_simulation);
                        print!("{}", s);
                        if let Some(statistics_path) = statistics_path_opt {
                            std::fs::write(statistics_path, s).unwrap();
                        }
                    }
                });
                *cancel.lock().unwrap() = true;
                std::process::exit(0);
            })
        };

        properties_window_main(
            Some(fluid_thread),
            cancel.clone(),
            restart.clone(),
            properties,
            visualization_params_arc,
        );
    } else if let Some(subcmd_matches) = matches.subcommand_matches("image") {
        let image_export_config_file: Vec<String> = subcmd_matches
            .values_of("IMAGE_EXPORT_CONFIG_FILES")
            .unwrap()
            .flat_map(|x| x.split(','))
            .map(String::from)
            .collect();
        animation::export_simulation_image(image_export_config_file);
    } else if let Some(subcmd_matches) = matches.subcommand_matches("generate-split-patterns") {
        let yaml_path = subcmd_matches.value_of("OUTPUT_YAML").unwrap();
        let split_patterns = generate_split_patterns(60);
        println!("Writing to file `{}`...", yaml_path);
        write_split_patterns_to_file(Path::new(yaml_path), &split_patterns);
        println!("SUCCESS DONE!");
    } else {
        unreachable!()
    }
}

fn fluid_main(
    cancel: Arc<Mutex<bool>>,
    restart: Arc<Mutex<bool>>,
    properties: Arc<Mutex<SimulationParams>>,
    scene_config: &SceneConfig,
    visualization_params_arc: Arc<Mutex<VisualizationParams>>,
    max_seconds: Option<FT>,
    counters_enabled: bool,
) -> Result<FluidSimulation<DimensionUtils2d, 2>, String> {
    let export_video = false;

    {
        let mut simulation_params = properties.lock().unwrap();
        init_simulation_params(&mut simulation_params, scene_config);
    }

    let split_patterns = load_split_patterns_from_file(Path::new("./split-patterns.yaml"));

    let mut fluid_simulation = init_fluid_sim(
        properties.lock().unwrap().clone(),
        scene_config,
        split_patterns,
        counters_enabled,
    );

    let mut _xmouse = 0;
    let mut _ymouse = 0;

    let mut total_duration: Duration = Duration::from_nanos(0);
    let mut total_number_of_frames = 0;

    let mut frame_number = 0;

    let mut window = SimulationWindow::new()?;
    let mut video_encoder = if export_video {
        Some(VideoEncoder::new(
            "video.mkv",
            window.get_window_width(),
            window.get_window_height(),
        ))
    } else {
        None
    };

    let export_vtk_data = false;
    let mut vtk_exporter;
    if export_vtk_data {
        vtk_exporter = Some(VtkExporter::new("/tmp/adaptive-sph-data", "my-sph"));
    } else {
        vtk_exporter = None;
    }

    let mut simulation_failed = false;

    loop {
        let a = std::time::Instant::now();
        if *cancel.lock().unwrap() {
            break;
        }

        if *restart.lock().unwrap() {
            fluid_simulation = init_fluid_sim(
                properties.lock().unwrap().clone(),
                scene_config,
                fluid_simulation.split_patterns,
                counters_enabled,
            );
            *restart.lock().unwrap() = false;
            simulation_failed = false;
        }

        let mut simulation_params: SimulationParams = { (*properties.lock().unwrap()).clone() };
        let visualization_params: VisualizationParams = { (*visualization_params_arc.lock().unwrap()).clone() };
        let continue_simulation = window.present(
            &fluid_simulation.particles,
            &fluid_simulation.neighs,
            &fluid_simulation.boundary_handler,
            &mut simulation_params,
            visualization_params,
            simulation_failed,
        )?;
        if !continue_simulation {
            break;
        }
        let b = std::time::Instant::now();

        if !simulation_failed {
            println!("rendering msecs {}", (b - a).as_secs_f32() * 1000.);

            // perform the main loop
            let a = std::time::Instant::now();
            simulation_failed = catch_unwind(AssertUnwindSafe(|| {
                fluid_simulation.single_step(simulation_params);
                if let Some(vtk_exporter) = &mut vtk_exporter {
                    vtk_exporter.add_snapshot(
                        fluid_simulation.time,
                        &fluid_simulation.particles,
                        &fluid_simulation.boundary_handler,
                        simulation_params,
                    );
                }
            }))
            .is_err();
            let b = std::time::Instant::now();

            total_duration += b - a;
            total_number_of_frames = total_number_of_frames + 1;

            let num_boundary_particles_opt: Option<usize> = match &fluid_simulation.boundary_handler {
                BoundaryHandler::ParticleBasedBoundaryHandler(bh) => Some(bh.num_boundary_particles()),
                _ => None,
            };
            let num_boundary_particles_str = match num_boundary_particles_opt {
                Some(num_boundary_particles) => format!("{}k fluid particles ", num_boundary_particles / 1000),
                None => format!(""),
            };

            println!(
                "{:05}: {} fluid particles {}{}msec ({}msec AVG)\n",
                frame_number,
                fluid_simulation.num_fluid_particles(),
                num_boundary_particles_str,
                (b - a).as_secs_f32() * 1000.,
                (total_duration / total_number_of_frames).as_secs_f32() * 1000.
            );

            frame_number += 1;
        }

        // if simulation_time >= 5.0 {
        //     break;
        // }

        if let Some(video_encoder) = &mut video_encoder {
            video_encoder.write_canvas(window.get_canvas());
        }

        if let Some(max_seconds) = max_seconds {
            if fluid_simulation.time >= max_seconds {
                break;
            }
        }

        // if frame_number == 10000 {
        //     break;
        // }
        //std::thread::sleep(Duration::from_secs_f32(0.1));

        // The rest of the game loop goes here...
    }

    Ok(fluid_simulation)
}
