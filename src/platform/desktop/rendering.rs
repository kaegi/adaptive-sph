use std::time::Instant;

use sdl2::{
    event::Event,
    gfx::primitives::DrawRenderer,
    keyboard::Keycode,
    mouse::MouseButton,
    pixels::{Color, PixelFormatEnum},
    rect::Rect,
    render::{Texture, TextureCreator},
    video::WindowContext,
    EventPump,
};

use cairo::{Format, ImageSurface};

use crate::{
    boundary_handler::BoundaryHandler,
    colors::get_color_for_particle,
    floating_type_mod::FT,
    neighborhood_search::NeighborhoodCache,
    platform::desktop::{animation::cairo_renderer::render2d, video_encoder::capture_foto},
    sdf::Sdf,
    simulation_parameters::SimulationParams,
    sph_kernels::{support_radius_single, DimensionUtils, DimensionUtils2d, ParticleSizes, PARTICLE_SIZES},
    vec2f, DrawShape, ParticleVec, SimulationVisualizer, VisualizationParams, VisualizedAttribute, V, V2, VF, M,
};

pub struct SimulationWindow {
    event_pump: EventPump,
    canvas: sdl2::render::WindowCanvas,
    zoom: FT,
    offset: V2,
    mouse: V2,
    drag: bool,
    pull_fluid: bool,
    start_instant: Instant,
    // the sdl2_texture has a implicit lifetime reference to _sdl2_texture_creator
    _sdl2_texture_creator: TextureCreator<WindowContext>,
    sdl2_texture: Texture,
    cairo_surface: ImageSurface,
}

impl SimulationVisualizer<DimensionUtils2d, 2> for SimulationWindow {
    fn present(
        &mut self,
        particles: &ParticleVec<2>,
        neighs: &NeighborhoodCache,
        boundary_handler: &BoundaryHandler<DimensionUtils2d, 2>,
        simulation_params: &mut SimulationParams,
        visualization_params: VisualizationParams,
        simulation_failed: bool,
    ) -> Result<bool, String> {
        type DU = DimensionUtils2d;
        const D: usize = 2;

        let (window_width, window_height) = self.canvas.window().size();

        let canvas = &mut self.canvas;

        for event in self.event_pump.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => return Ok(false),
                Event::KeyDown {
                    keycode: Some(Keycode::F12),
                    ..
                } => {
                    capture_foto("./photo", &canvas);
                }
                Event::MouseMotion { xrel, yrel, x, y, .. } => {
                    if self.drag {
                        self.offset.x += xrel as FT;
                        self.offset.y += yrel as FT;
                    }
                    self.mouse.x = x as FT;
                    self.mouse.y = y as FT;
                }
                Event::MouseButtonDown { mouse_btn, .. } => {
                    if mouse_btn == MouseButton::Left {
                        self.drag = true;
                    }
                    if mouse_btn == MouseButton::Right {
                        self.pull_fluid = true;
                    }
                }
                Event::MouseButtonUp { mouse_btn, .. } => {
                    if mouse_btn == MouseButton::Left {
                        self.drag = false;
                    }
                    if mouse_btn == MouseButton::Right {
                        self.pull_fluid = false;
                    }
                }
                Event::MouseWheel { y, .. } => {
                    // _xmouse = x;
                    // _ymouse = y;
                    let step: FT = 1.1;
                    self.zoom *= step.powi(y);
                    self.offset *= step.powi(y);
                }
                _ => {}
            }
        }

        if self.pull_fluid {
            let mut x = self.mouse;
            x -= self.offset;
            x.y = window_height as FT - x.y;
            x.x += -(window_width as FT) * 0.5;
            x.y += -(window_height as FT) * 0.5;
            x /= self.zoom;
            simulation_params.pull_fluid_to = Some([x[0], x[1], 0.].into());
        } else {
            simulation_params.pull_fluid_to = None;
        }

        if visualization_params.draw_shape == DrawShape::Cairo {
            render2d(
                &particles.position,
                &particles.mass,
                simulation_params.rest_density,
                |i| {
                    get_color_for_particle::<DimensionUtils2d, 2>(
                        i,
                        particles,
                        simulation_params.clone(),
                        None,
                        neighs,
                        visualization_params,
                    )
                },
                boundary_handler,
                &self.cairo_surface,
                None,
                Some(&String::from("my title")),
                1.04,
            );

            self.cairo_surface.flush();
            let cairo_stride = self.cairo_surface.format().stride_for_width(window_width).unwrap() as usize;
            let cairo_data = self.cairo_surface.data().unwrap();

            self.sdl2_texture
                .with_lock(None, |buffer: &mut [u8], pitch: usize| {
                    assert_eq!(pitch, cairo_stride);
                    buffer.copy_from_slice(&cairo_data);
                    // for x in 0..window_width as usize {
                    //     for y in 0..window_height as usize {
                    //         let r: u8 = data[y * cairo_stride as usize + x * 4 + 0];
                    //         let g: u8 = data[y * cairo_stride as usize + x * 4 + 1];
                    //         let b: u8 = data[y * cairo_stride as usize + x * 4 + 2];
                    //         buffer[y * pitch + x * 4 + 0] = r;
                    //         buffer[y * pitch + x * 4 + 1] = g;
                    //         buffer[y * pitch + x * 4 + 2] = b;
                    //         // canvas.pixel(x as i16, y as i16, Color::RGB(r, g, b)).unwrap();
                    //     }
                    // }
                })
                .unwrap();

            canvas.copy(&self.sdl2_texture, None, None).unwrap();
            canvas.present();
            Ok(true)
        } else {
            if simulation_failed {
                canvas.set_draw_color(Color::RGB(40, 0, 0));
            } else {
                canvas.set_draw_color(Color::WHITE);
            }
            canvas.clear();
            canvas.set_draw_color(Color::WHITE);

            let zoom_factor: FT = self.zoom;

            let viewport_size = vec2f(canvas.viewport().width() as FT, canvas.viewport().height() as FT);

            let world_pos_to_screen_pos = |mut x: VF<2>| -> VF<2> {
                x *= zoom_factor;
                x.x -= -(window_width as FT) * 0.5;
                x.y -= -(window_height as FT) * 0.5;
                x.y = window_height as FT - x.y;
                x += self.offset;
                return x;
            };

            let world_pos_to_screen_pos_i = |x: VF<2>| -> (i32, i32) {
                let p = world_pos_to_screen_pos(x);
                (p.x as i32, p.y as i32)
            };

            if let Some(pull_fluid_to) = simulation_params.pull_fluid_to {
                let screen_pos = world_pos_to_screen_pos(vec2f(pull_fluid_to[0], pull_fluid_to[1]));
                let size = 0.03;
                canvas
                    .filled_circle(
                        screen_pos.x as i16,
                        screen_pos.y as i16,
                        (size * zoom_factor) as i16,
                        Color::RED,
                    )
                    .unwrap();

                let ms = self.start_instant.elapsed().as_millis();

                canvas
                    .filled_circle(
                        screen_pos.x as i16,
                        screen_pos.y as i16,
                        (size * zoom_factor * (1. - (ms % 500) as FT / 500.)) as i16,
                        Color::YELLOW,
                    )
                    .unwrap();

                canvas
                    .circle(
                        screen_pos.x as i16,
                        screen_pos.y as i16,
                        (size * zoom_factor) as i16,
                        Color::BLACK,
                    )
                    .unwrap();
            }

            let dot_draw_size: V<i32, 2> = [2, 2].into();

            match boundary_handler {
                BoundaryHandler::ParticleBasedBoundaryHandler(particle_boundary_handler) => {
                    for boundary_particle_id in 0..particle_boundary_handler.num_boundary_particles() {
                        let position = particle_boundary_handler.boundary_positions[boundary_particle_id];

                        canvas.set_draw_color(Color::RGB(100, 100, 100));

                        let screen_pos = world_pos_to_screen_pos(position);
                        canvas.fill_rect(Rect::new(
                            (screen_pos.x - dot_draw_size.x as FT).round() as i32,
                            (screen_pos.y - dot_draw_size.y as FT).round() as i32,
                            dot_draw_size.x as u32,
                            dot_draw_size.y as u32,
                        ))?;
                    }
                }
                BoundaryHandler::BoundaryWinchenbach2020(boundary_handler) => {
                    canvas.set_draw_color(Color::BLACK);
                    for sdf in &boundary_handler.sdf {
                        match sdf {
                            Sdf::Sdf2D(sdf2d) => {
                                for (a, b) in sdf2d.draw_lines() {
                                    canvas.draw_line(world_pos_to_screen_pos_i(a), world_pos_to_screen_pos_i(b))?;
                                }
                            }
                            Sdf::SdfPlane(sdf_plane) => {
                                let (a, b) = sdf_plane.get_two_points_with_distance(20.);
                                // for (a, b) in sdf2d.draw_lines() {
                                canvas.draw_line(world_pos_to_screen_pos_i(a), world_pos_to_screen_pos_i(b))?;
                                // }
                            }
                            _ => {
                                unimplemented!()
                            }
                        }
                    }
                }
                BoundaryHandler::NoBoundaryHandler(_) => {}
            }

            let num_particles = particles.position.len();

            let max_pressure = if visualization_params.visualized_attribute == VisualizedAttribute::Pressure {
                particles
                    .pressure
                    .iter()
                    .cloned()
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap()
            } else {
                0.
            };

            for i in 0..num_particles {
                let position = particles.position[i];
                // let radius = particles.mass[particle_id] / simulation_params.rest_density;

                let draw_color_ft = get_color_for_particle::<DU, D>(
                    i,
                    particles,
                    simulation_params.clone(),
                    Some(max_pressure),
                    neighs,
                    visualization_params,
                );

                let draw_color = Color::RGB(
                    (draw_color_ft[0] * 255.0) as u8,
                    (draw_color_ft[1] * 255.0) as u8,
                    (draw_color_ft[2] * 255.0) as u8,
                );

                /*let density_dev = ;
                if density_dev < 1. {
                    let x = density_dev.powi(3);
                    canvas.set_draw_color(Color::RGB((255. * x) as u8, (255. * x) as u8, 255));
                } else {
                    let x = (1. as FT / density_dev).powi(3);
                    canvas.set_draw_color(Color::RGB(255, (255. * x) as u8, (255. * x) as u8));
                }

                let pressure_val = FT::clamp(1.0 - particles.pressure[particle_id] / maximum_pressure, 0., 1.);
                canvas.set_draw_color(Color::RGB(
                    255,
                    (255. * pressure_val) as u8,
                    (255. * pressure_val) as u8,
                ));

                // pressure was clamped
                if particles.flag2[particle_id] {
                    canvas.set_draw_color(Color::GREEN);
                }

                if let LevelEstimationState::FluidSurface(level_estimate) = particles.level_estimation[particle_id] {
                    canvas.set_draw_color(Color::RGB(0, 255 - (FT::min(-level_estimate * 9., 230.)) as u8, 100));
                }*/

                let radius = <DU as DimensionUtils<D>>::sphere_volume_to_radius(
                    particles.mass[i] / simulation_params.rest_density,
                );

                let screen_pos = world_pos_to_screen_pos(position);

                let screen_radius = radius * zoom_factor;

                if screen_pos.x + screen_radius < 0.
                    || screen_pos.y + screen_radius < 0.
                    || screen_pos.x - screen_radius >= viewport_size.x
                    || screen_pos.y - screen_radius >= viewport_size.y
                {
                    continue;
                }

                match visualization_params.draw_shape {
                    DrawShape::FilledCircle => {
                        canvas
                            .filled_circle(
                                screen_pos.x.round() as i16,
                                screen_pos.y.round() as i16,
                                (radius * zoom_factor) as i16,
                                draw_color,
                            )
                            .unwrap();
                    }
                    DrawShape::FilledCircleWithBorder => {
                        canvas
                            .filled_circle(
                                screen_pos.x.round() as i16,
                                screen_pos.y.round() as i16,
                                (radius * zoom_factor) as i16,
                                draw_color,
                            )
                            .unwrap();
                        canvas
                            .circle(
                                screen_pos.x.round() as i16,
                                screen_pos.y.round() as i16,
                                (radius * zoom_factor) as i16,
                                Color::BLACK,
                            )
                            .unwrap();
                    }
                    DrawShape::FilledCircleWithAABorder => {
                        canvas
                            .filled_circle(
                                screen_pos.x.round() as i16,
                                screen_pos.y.round() as i16,
                                (radius * zoom_factor) as i16,
                                draw_color,
                            )
                            .unwrap();
                        canvas
                            .aa_circle(
                                screen_pos.x.round() as i16,
                                screen_pos.y.round() as i16,
                                (radius * zoom_factor) as i16,
                                Color::BLACK,
                            )
                            .unwrap();
                    }
                    DrawShape::Circle => {
                        canvas
                            .circle(
                                screen_pos.x.round() as i16,
                                screen_pos.y.round() as i16,
                                (radius * zoom_factor) as i16,
                                draw_color,
                            )
                            .unwrap();
                        // canvas
                        //     .aa_circle(
                        //         screen_pos.x.round() as i16,
                        //         screen_pos.y.round() as i16,
                        //         (radius * zoom_factor) as i16,
                        //         draw_color,
                        //     )
                        //     .unwrap();
                    }
                    DrawShape::Dot => {
                        canvas.set_draw_color(draw_color);
                        canvas.fill_rect(Rect::new(
                            (screen_pos.x - dot_draw_size.x as FT).round() as i32,
                            (screen_pos.y - dot_draw_size.y as FT).round() as i32,
                            dot_draw_size.x as u32,
                            dot_draw_size.y as u32,
                        ))?;
                    }
                    DrawShape::Cairo => unreachable!(),
                }
            }

            if visualization_params.draw_support_radius {
                let avg_sr: FT = match PARTICLE_SIZES {
                    ParticleSizes::Adaptive => {
                        let sum: FT = (0..num_particles)
                            .into_iter()
                            .map(|i| support_radius_single::<DU, D>(&particles.h2, i, simulation_params.clone()))
                            .sum();
                        sum / num_particles as FT
                    }
                    ParticleSizes::Uniform => {
                        simulation_params.h * <DU as DimensionUtils<D>>::support_radius_by_smoothing_length()
                    }
                };

                // draw support radius
                for i in 0..num_particles {
                    if support_radius_single::<DU, D>(&particles.h2, i, simulation_params.clone()) > avg_sr * 0.9
                        && i % (num_particles / 20) != 0
                    {
                        continue;
                    }

                    // draw the kernel support radius around it
                    let radius = support_radius_single::<DU, D>(&particles.h2, i, simulation_params.clone());
                    // let radius = smoothing_length_single(&particles.h2, i, simulation_params);
                    let screen_pos = world_pos_to_screen_pos(particles.position[i]);
                    canvas
                        .circle(
                            screen_pos.x as i16,
                            screen_pos.y as i16,
                            (radius * zoom_factor) as i16,
                            Color::WHITE,
                        )
                        .unwrap();
                }
            }

            canvas.present();

            Ok(true)
        }
    }
}

impl SimulationWindow {
    pub fn new() -> Result<Self, String> {
        let window_height: u32 = 800;
        let window_width: u32 = 800;

        let sdl_context = sdl2::init()?;
        let video_subsystem = sdl_context.video()?;
        let window = video_subsystem
            .window("adaptive-sph fluid", window_width, window_height)
            .position_centered()
            .opengl()
            .build()
            .map_err(|e| e.to_string())?;

        let canvas = window.into_canvas().build().map_err(|e| e.to_string())?;
        let event_pump = sdl_context.event_pump()?;

        let zoom = window_width as FT / (2.1 * M);

        let sdl2_texture_creator = canvas.texture_creator();

        let sdl2_texture = sdl2_texture_creator
            .create_texture_streaming(PixelFormatEnum::ARGB8888, window_width, window_height)
            .unwrap();

        let cairo_format = Format::Rgb24;
        let cairo_stride = cairo_format.stride_for_width(window_width).unwrap() as usize;
        let data: Vec<u8> = vec![0; window_height as usize * cairo_stride];
        let cairo_surface = ImageSurface::create_for_data(
            data,
            cairo_format,
            window_width as i32,
            window_height as i32,
            cairo_stride as i32,
        )
        .unwrap();

        Ok(SimulationWindow {
            event_pump,
            canvas,
            zoom,
            drag: false,
            offset: V2::zeros(),
            mouse: V2::zeros(),
            pull_fluid: false,
            start_instant: Instant::now(),
            _sdl2_texture_creator: sdl2_texture_creator,
            sdl2_texture,
            cairo_surface,
        })
    }

    pub fn get_window_width(&self) -> u32 {
        self.canvas.window().size().0
    }

    pub fn get_window_height(&self) -> u32 {
        self.canvas.window().size().1
    }

    pub fn get_canvas(&self) -> &sdl2::render::WindowCanvas {
        &self.canvas
    }
}
