use cairo::{Context, FontSlant, FontWeight, ImageSurface, LinearGradient};
use std::f64::consts::TAU;

use crate::{
    boundary_handler::BoundaryHandler,
    color_map::ColorMap,
    floating_type_mod::FT,
    sdf::Sdf,
    sph_kernels::{DimensionUtils, DimensionUtils2d},
    V, VF,
};

pub struct Legend {
    pub color_map: ColorMap,
    pub text_right: bool,
    pub only_min_max: bool,
}

pub fn render2d(
    pos: &[VF<2>],
    mass: &[FT],
    rest_density: FT,
    particle_color: impl Fn(usize) -> V<f64, 3>,
    boundary_handler: &BoundaryHandler<DimensionUtils2d, 2>,
    surface: &ImageSurface,
    legend_opt: Option<Legend>,
    title_opt: Option<&String>,
    zoom_out: f64,
) {
    let img_width = surface.width();
    let img_height = surface.height();

    let context = Context::new(&surface).expect("Couldn't create cairo context!");

    // paint canvas white
    context.set_source_rgb(1.0, 1., 1.);
    context.paint().unwrap(); // draw 100 random black lines
                              // for _i in 0..100 {
                              //     let x = rand::random::<f64>() * 600.0;
                              //     let y = rand::random::<f64>() * 600.0;
                              //     context.line_to(x, y);
                              // }
                              // context.stroke().unwrap();

    let scene_width = 2.;
    // let zoom_out = 1.04;
    let scale = i32::min(img_width, img_height) as f64 / (scene_width * zoom_out);

    context.translate(img_width as f64 / 2., img_height as f64 / 2.);
    context.scale(scale, scale);
    context.scale(1., -1.);

    type DU = DimensionUtils2d;
    const D: usize = 2;

    match boundary_handler {
        BoundaryHandler::ParticleBasedBoundaryHandler(particle_boundary_handler) => {
            for _boundary_particle_id in 0..particle_boundary_handler.num_boundary_particles() {
                // let position = particle_boundary_handler.boundary_positions[boundary_particle_id];

                // context.set_source_rgb(0.4, 0.4, 0.4);

                unimplemented!()
            }
        }
        BoundaryHandler::BoundaryWinchenbach2020(boundary_handler) => {
            context.set_source_rgb(0., 0., 0.);
            for sdf in &boundary_handler.sdf {
                match sdf {
                    Sdf::Sdf2D(sdf2d) => {
                        context.set_line_width(5. / 1000.);
                        for (a, b) in sdf2d.draw_lines() {
                            context.move_to(a.x as f64, a.y as f64);
                            context.line_to(b.x as f64, b.y as f64);
                            context.stroke().unwrap();
                        }
                    }
                    Sdf::SdfPlane(sdf_plane) => {
                        context.set_line_width(5. / 1000.);
                        let (a, b) = sdf_plane.get_two_points_with_distance(5.);
                        context.move_to(a.x as f64, a.y as f64);
                        context.line_to(b.x as f64, b.y as f64);
                        context.stroke().unwrap();
                    }
                    _ => {
                        unimplemented!()
                    }
                }
            }
        }
        BoundaryHandler::NoBoundaryHandler(_) => {}
    }

    for i in 0..pos.len() {
        let radius = <DU as DimensionUtils<D>>::sphere_volume_to_radius(mass[i] / rest_density);
        let pos = pos[i];
        let color: V<f64, 3> = particle_color(i);

        context.set_source_rgb(color.x, color.y, color.z);
        context.arc(pos.x as f64, pos.y as f64, radius as f64, 0., TAU);
        context.fill().unwrap();

        context.set_source_rgb(0., 0., 0.);
        context.set_line_width(radius as f64 * 0.1);
        context.arc(pos.x as f64, pos.y as f64, radius as f64, 0., TAU);
        context.stroke().unwrap();

        // if pressures[i] > 10000. {
        //     println!("POSITION {} {:?}", i, pos);
        //     context.set_source_rgb(0., 0., 0.);
        //     context.set_line_width(radius as f64 * 0.5);
        //     context.arc(pos.x as f64, pos.y as f64, radius as f64 * 5., 0., TAU);
        //     context.stroke().unwrap();
        // }
    }

    if let Some(legend) = legend_opt {
        let legend_min: V<f64, 2> = [img_width as f64 * 0.83, img_height as f64 * 0.5].into();
        let legend_size: V<f64, 2> = [img_width as f64 * 0.07, img_height as f64 * 0.3].into();

        let gradient: LinearGradient = LinearGradient::new(0., legend_min.y, 0., legend_min.y + legend_size.y);
        let min_value: f64 = legend.color_map.color_stops().first().unwrap().0 as f64;
        let max_value: f64 = legend.color_map.color_stops().last().unwrap().0 as f64;
        for (v, color) in legend.color_map.color_stops() {
            let interp = ((*v as f64 - min_value) / (max_value - min_value)) as f64;
            gradient.add_color_stop_rgb(interp, color[0] as f64, color[1] as f64, color[2] as f64);
        }
        context.identity_matrix();
        context.translate(0., img_height as f64);
        context.scale(1., -1.);

        context.set_source(&gradient).unwrap();
        context.rectangle(legend_min.x, legend_min.y, legend_size.x, legend_size.y);
        context.fill().unwrap();

        context.set_source_rgb(0., 0., 0.);
        context.set_line_width(5.);
        context.rectangle(legend_min.x, legend_min.y, legend_size.x, legend_size.y);
        context.stroke().unwrap();

        context.identity_matrix();
        context.select_font_face("Akaash", FontSlant::Normal, FontWeight::Normal);
        context.set_font_size(img_height as f64 * 0.04);
        let indicator_line_width = img_width as f64 * 0.01;

        let stops: Vec<f64>;
        if legend.only_min_max {
            stops = vec![min_value as f64, max_value as f64];
        } else {
            stops = legend
                .color_map
                .color_stops()
                .into_iter()
                .map(|(v, _)| *v as f64)
                .collect();
        }

        for v in stops {
            let interp = ((v - min_value) / (max_value - min_value)) as f64;
            let ycenter = img_height as f64 - (legend_min.y + interp * legend_size.y);
            let s = &format!("{}", ((v * 1000.).round() as i32) as f64 / 1000.);
            context.set_source_rgb(0., 0., 0.);
            let text_extents = context.text_extents(s).unwrap();

            if legend.text_right {
                let tx = legend_min.x + legend_size.x + indicator_line_width + img_width as f64 * 0.008;
                let ty = ycenter + text_extents.height() * 0.5;
                // context.set_source_rgb(1., 1., 1.);
                // context.rectangle(tx + text_extents.x_bearing, ty + text_extents.y_bearing, text_extents.width, text_extents.height);
                // context.fill().unwrap();

                context.set_source_rgb(0., 0., 0.);
                context.move_to(tx, ty);
                context.show_text(s).unwrap();

                context.move_to(legend_min.x + legend_size.x, ycenter);
                context.line_to(legend_min.x + legend_size.x + indicator_line_width, ycenter);
                context.stroke().unwrap();
            } else {
                let tx = legend_min.x - text_extents.width() - indicator_line_width - img_width as f64 * 0.008;
                let ty = ycenter + text_extents.height() * 0.5;
                // context.set_source_rgb(1., 1., 1.);
                // context.rectangle(tx + text_extents.x_bearing, ty + text_extents.y_bearing, text_extents.width, text_extents.height);
                // context.fill().unwrap();

                context.set_source_rgb(0., 0., 0.);
                context.move_to(tx, ty);
                context.show_text(s).unwrap();

                context.move_to(legend_min.x, ycenter);
                context.line_to(legend_min.x - indicator_line_width, ycenter);
                context.stroke().unwrap();
            }
        }
    }

    if let Some(title) = title_opt {
        let title2 = title.replace("#p", &format!("{}", pos.len()));

        context.identity_matrix();
        context.select_font_face("Akaash", FontSlant::Normal, FontWeight::Normal);
        context.set_font_size(img_width as f64 * 0.048);

        let text_extents = context.text_extents(&title2).unwrap();

        let tx = img_width as f64 * 0.02;
        let ty = img_height as f64 * 0.01 + text_extents.height();

        context.set_line_width(img_height as f64 * 0.012);
        context.set_source_rgb(1., 1., 1.);
        context.move_to(tx, ty);
        context.text_path(&title2);
        context.stroke().unwrap();

        context.set_source_rgb(0., 0., 0.);
        context.move_to(tx, ty);
        context.show_text(&title2).unwrap();
    }
}
