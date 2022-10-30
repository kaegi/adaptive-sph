use std::{
    collections::HashMap,
    fs::{create_dir_all, remove_dir_all},
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use cairo::{Format, ImageSurface};
use serde::{Deserialize, Serialize};
use serde_yaml::{from_value, Value};
use std::fs::File;

use crate::{
    colors::{get_color_for_particle, get_color_map, get_color_map_for_pressure},
    floating_type_mod::FT,
    init_fluid_sim, init_simulation_params, load_split_patterns_from_file,
    platform::desktop::animation::cairo_renderer::Legend,
    simulation_parameters::SimulationParams,
    sph_kernels::DimensionUtils2d,
    write_statistics, SceneConfig, VisualizationParams, VisualizedAttribute, VF,
};

use self::cairo_renderer::render2d;

pub mod cairo_renderer;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ImageExportConfig {
    time: FT,
    video_start_time: Option<FT>,
    video_fps: Option<FT>,
    video_speed: Option<FT>,
    zoom_out: Option<f64>,
    #[serde(default = "bool::default")]
    interpolated: bool,
    #[serde(default = "bool::default")]
    no_legend: bool,
    #[serde(default = "bool::default")]
    legend_text_right: bool,
    #[serde(default = "bool::default")]
    legend_only_min_max: bool,
    title: Option<String>,
    config_path: String,
    scene: Option<SceneConfig>,
    scene_file: Option<String>,
    #[serde(default)]
    update_attributes: HashMap<String, serde_yaml::Value>,
    visualization_params: VisualizationParams,
    png_file: String,
    output_stats: Option<bool>,
    panic_on_end: Option<bool>,
    export_when_mii_negative: Option<bool>,
    video_img_dir: Option<String>,
    image_width: Option<i32>,
    image_height: Option<i32>,
}

pub fn export_simulation_image(image_export_config_paths: Vec<String>) {
    type DU = DimensionUtils2d;
    const D: usize = 2;

    for image_export_config_path in image_export_config_paths {
        let path = std::fs::canonicalize(PathBuf::from(image_export_config_path)).unwrap();
        let path2 = path.clone();
        let dir = path2.parent().unwrap().clone();

        let image_export_config_yaml = std::fs::read_to_string(path).expect("failed export config file");
        let image_export_configs: Vec<ImageExportConfig> =
            serde_yaml::from_str(&image_export_config_yaml).expect("failed parsing export config file");

        for image_export_config in image_export_configs {
            let params_yaml = std::fs::read_to_string(dir.join(image_export_config.config_path))
                .expect("failed reading parameter file");
            let mut simulation_params_serde: Value =
                serde_yaml::from_str(&params_yaml).expect("failed parsing simulation config file");

            let scene = match (image_export_config.scene, image_export_config.scene_file) {
                (None, None) => panic!("expected either 'scene' or 'scene_file'"),
                (Some(_), Some(_)) => panic!("expected either 'scene' or 'scene_file'. Not both!"),
                (Some(scene), None) => scene,
                (None, Some(scene_file_path)) => {
                    let scene_yaml =
                        std::fs::read_to_string(dir.join(scene_file_path)).expect("failed reading scene file");
                    serde_yaml::from_str(&scene_yaml).expect("failed parsing scene config file")
                }
            };

            for (k, v) in image_export_config.update_attributes.into_iter() {
                let mapping = simulation_params_serde
                    .as_mapping_mut()
                    .expect("cannot get parsed simulation parameters as mapping");
                *mapping
                    .get_mut(&Value::String(k.clone()))
                    .unwrap_or_else(|| panic!("not able to find attribute {}", k)) = v;
            }

            let mut simulation_params: SimulationParams =
                from_value(simulation_params_serde).expect("failed to unpack SimulationParams");

            init_simulation_params(&mut simulation_params, &scene);

            let split_patterns = load_split_patterns_from_file(Path::new("./split-patterns.yaml"));
            let mut fluid_simulation = init_fluid_sim(simulation_params, &scene, split_patterns, true);

            #[derive(Copy, Clone, Debug)]
            struct VideoInfo {
                start: FT,
                end: FT,
                fps: FT,
                speed: FT,
            }
            let video_opt: Option<VideoInfo> = image_export_config.video_start_time.map(|start| VideoInfo {
                start: start,
                end: image_export_config.time,
                fps: image_export_config.video_fps.unwrap_or(60.),
                speed: image_export_config.video_speed.unwrap_or(1.),
            });
            let mut video_frame_counter = 0;

            let mut time_for_next_export = match video_opt {
                Some(video) => video.start,
                None => image_export_config.time,
            };

            let video_img_dir: PathBuf = if let Some(video_img_dir) = image_export_config.video_img_dir.clone() {
                dir.join(video_img_dir)
            } else {
                PathBuf::from("/tmp/sph")
            };

            if video_opt.is_some() {
                println!("re-create video image dir: {:?}", video_img_dir);
                remove_dir_all(video_img_dir.clone()).ok();
                create_dir_all(video_img_dir.clone()).unwrap();
            }

            'simulation_loop: loop {
                let time_before_step = fluid_simulation.time;
                let position_before_step = fluid_simulation.particles.position.clone();
                let dt = fluid_simulation.single_step_without_adaptivity(simulation_params);

                if image_export_config.panic_on_end == Some(true) && fluid_simulation.time > image_export_config.time {
                    panic!(">>>>>>>>>>>> REACHED END BEFORE EXPORT <<<<<<<<<<<<");
                }

                // let mut export_image = fluid_simulation.time > time_for_next_export;
                // if Some(true) == image_export_config.export_when_mii_negative {
                //     export_image |= fluid_simulation.particles.aii.iter().any(|&p| p < 0.);
                // }

                while time_for_next_export <= fluid_simulation.time {
                    let img_width = image_export_config.image_width.unwrap_or(2000);
                    let img_height = image_export_config.image_height.unwrap_or(2000);
                    let surface = ImageSurface::create(Format::ARgb32, img_width, img_height)
                        .expect("Couldn't create a surface!");

                    let legend_opt = if !image_export_config.no_legend {
                        let color_map = if image_export_config.visualization_params.visualized_attribute
                            == VisualizedAttribute::Pressure
                        {
                            get_color_map_for_pressure::<DU, D>(
                                image_export_config.visualization_params.visualized_attribute,
                                simulation_params,
                                fluid_simulation.particles.pressure.iter().cloned().fold(0., FT::max),
                            )
                        } else {
                            get_color_map::<DU, D>(
                                image_export_config.visualization_params.visualized_attribute,
                                simulation_params,
                            )
                            .unwrap()
                        };

                        Some(Legend {
                            color_map,
                            text_right: image_export_config.legend_text_right,
                            only_min_max: image_export_config.legend_only_min_max,
                        })
                    } else {
                        None
                    };

                    let mut max_pressure = None;
                    if image_export_config.visualization_params.visualized_attribute == VisualizedAttribute::Pressure {
                        max_pressure =
                            Some(fluid_simulation.particles.pressure.iter().cloned().fold(0., FT::max) * 0.9);
                        // max_pressure = Some(10000.);
                    }

                    let mut render_positions;
                    if video_opt.is_some() {
                        let pos_interpolation =
                            (time_for_next_export - time_before_step) / (fluid_simulation.time - time_before_step);
                        if pos_interpolation < 0. {
                            panic!(
                                "negative interpolation {} (export {} between {} and {})",
                                pos_interpolation, time_for_next_export, fluid_simulation.time, time_before_step
                            );
                        }
                        assert!(pos_interpolation <= 1.);
                        render_positions = vec![VF::<2>::zeros(); fluid_simulation.particles.position.len()];

                        for i in 0..fluid_simulation.particles.position.len() {
                            render_positions[i] = pos_interpolation * fluid_simulation.particles.position[i]
                                + (1. - pos_interpolation) * position_before_step[i];
                        }
                    } else {
                        render_positions = fluid_simulation.particles.position.clone();
                    }

                    render2d(
                        &render_positions,
                        &fluid_simulation.particles.mass,
                        simulation_params.rest_density,
                        |i| {
                            get_color_for_particle::<DimensionUtils2d, 2>(
                                i,
                                &fluid_simulation.particles,
                                simulation_params,
                                max_pressure,
                                &fluid_simulation.neighs,
                                image_export_config.visualization_params,
                            )
                        },
                        &fluid_simulation.boundary_handler,
                        &surface,
                        legend_opt,
                        image_export_config.title.as_ref(),
                        image_export_config.zoom_out.unwrap_or(1.04),
                    );

                    let output_file_path: PathBuf = match video_opt {
                        Some(_video) => video_img_dir.join(format!("file-{:06}.png", video_frame_counter)),
                        None => dir.join(&image_export_config.png_file),
                    };
                    let mut file = File::create(output_file_path).expect("Couldn't create file");
                    surface.write_to_png(&mut file).expect("Couldn't write to png");

                    if let Some(video) = video_opt {
                        video_frame_counter = video_frame_counter + 1;
                        time_for_next_export += 1. / video.fps * video.speed;
                        if fluid_simulation.time > video.end {
                            Command::new("ffmpeg")
                                .args([
                                    "-y",
                                    "-framerate",
                                    (video.fps.round() as i32).to_string().as_str(),
                                    "-pattern_type",
                                    "glob",
                                    "-i",
                                    video_img_dir.join("*.png").to_str().unwrap(),
                                    "-c:v",
                                    "libx264",
                                    "-pix_fmt",
                                    "yuv420p",
                                    dir.join(&image_export_config.png_file).to_str().unwrap(),
                                ])
                                .stdin(Stdio::null())
                                .stdout(Stdio::inherit())
                                .stderr(Stdio::inherit())
                                .output()
                                .expect("failed to execute process");

                            break 'simulation_loop;
                        }
                    } else {
                        break 'simulation_loop;
                    }
                }

                fluid_simulation.single_step_adaptivity(simulation_params, dt);
            }

            if image_export_config.output_stats == Some(true) {
                let fp = dir.join(format!("{}.stat", image_export_config.png_file));
                let mut file = File::create(fp).expect("Couldn't create file");
                let stats = write_statistics(&fluid_simulation);
                file.write_all(stats.as_bytes()).unwrap();
                // file.write_fmt(format_args!(
                //     "num_particles: {}",
                //     fluid_simulation.num_fluid_particles()
                // ))
                // .unwrap();
            }
        }
    }
}
