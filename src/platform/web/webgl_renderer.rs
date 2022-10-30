use std::sync::Arc;
use std::sync::Mutex;

use nalgebra::Matrix2x3;
use nalgebra::Matrix3;
use nalgebra::Vector3;
use once_cell::sync::Lazy;
use send_wrapper::SendWrapper;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::MouseEvent;
use web_sys::Touch;
use web_sys::TouchList;
use web_sys::WebGlBuffer;
use web_sys::WebGlFramebuffer;
use web_sys::WebGlTexture;
use web_sys::WebGlVertexArrayObject;
use web_sys::{WebGl2RenderingContext, WebGlProgram, WebGlShader};

use super::web_loop::InputState;

use crate::boundary_handler::BoundaryHandler;
use crate::colors::get_color_for_particle;
use crate::floating_type_mod::FT;
use crate::platform::web::web_loop::GlobalState;
use crate::sdf::Sdf;
use crate::sph_kernels::DimensionUtils;
use crate::sph_kernels::DimensionUtils2d;
use crate::DrawShape;
use crate::VisualizationParams;
use crate::V;
use crate::VF;

struct FramebufferData {
    fb: WebGlFramebuffer,
    tex: WebGlTexture,
    width: u32,
    height: u32,
}

impl FramebufferData {
    fn new(context: &WebGl2RenderingContext, width: u32, height: u32) -> Result<Self, JsValue> {
        let tex = context.create_texture().unwrap();
        context.bind_texture(WebGl2RenderingContext::TEXTURE_2D, Some(&tex));

        Self::create_texture_with_size(context, width, height)?;

        let level = 0;
        let fb = context.create_framebuffer().unwrap();
        context.bind_framebuffer(WebGl2RenderingContext::FRAMEBUFFER, Some(&fb));
        context.framebuffer_texture_2d(
            WebGl2RenderingContext::FRAMEBUFFER,
            WebGl2RenderingContext::COLOR_ATTACHMENT0,
            WebGl2RenderingContext::TEXTURE_2D,
            Some(&tex),
            level,
        );

        Ok(FramebufferData { fb, tex, width, height })
    }

    fn create_texture_with_size(context: &WebGl2RenderingContext, width: u32, height: u32) -> Result<(), JsValue> {
        // define size and format of level 0
        let level = 0;
        let internal_format = WebGl2RenderingContext::RGBA;
        let border = 0;
        let format = WebGl2RenderingContext::RGBA;
        let type_ = WebGl2RenderingContext::UNSIGNED_BYTE;
        let data = None;
        context.tex_image_2d_with_i32_and_i32_and_i32_and_format_and_type_and_opt_u8_array(
            WebGl2RenderingContext::TEXTURE_2D,
            level,
            internal_format as i32,
            width as i32,
            height as i32,
            border,
            format,
            type_,
            data,
        )?;

        // set the filtering so we don't need mips
        context.tex_parameteri(
            WebGl2RenderingContext::TEXTURE_2D,
            WebGl2RenderingContext::TEXTURE_MIN_FILTER,
            WebGl2RenderingContext::LINEAR as i32,
        );
        context.tex_parameteri(
            WebGl2RenderingContext::TEXTURE_2D,
            WebGl2RenderingContext::TEXTURE_WRAP_S,
            WebGl2RenderingContext::CLAMP_TO_EDGE as i32,
        );
        context.tex_parameteri(
            WebGl2RenderingContext::TEXTURE_2D,
            WebGl2RenderingContext::TEXTURE_WRAP_T,
            WebGl2RenderingContext::CLAMP_TO_EDGE as i32,
        );

        Ok(())
    }

    fn ensure_size_and_bind_fb(
        &self,
        context: &WebGl2RenderingContext,
        width: u32,
        height: u32,
    ) -> Result<(), JsValue> {
        if width != self.width || height != self.height {
            context.bind_texture(WebGl2RenderingContext::TEXTURE_2D, Some(&self.tex));
            Self::create_texture_with_size(context, width, height)?;
        }

        context.bind_framebuffer(WebGl2RenderingContext::FRAMEBUFFER, Some(&self.fb));
        context.viewport(0, 0, width as i32, height as i32);
        Ok(())
    }
}
struct Rendering {
    circle_buffer: WebGlBuffer,
    circle_vao: WebGlVertexArrayObject,
    circle_program: WebGlProgram,
    circle_vertices: Vec<f32>,

    line_buffer: WebGlBuffer,
    line_vao: WebGlVertexArrayObject,
    line_program: WebGlProgram,
    line_vertices: Vec<f32>,

    context: WebGl2RenderingContext,

    transform_matrix: Matrix2x3<f32>,
    inv_transform_matrix: Matrix2x3<f32>,

    touch_distance: FT,
    avg_touch_pos: VF<2>,

    move_canvas: bool,

    view_offset: VF<2>,
    zoom: FT,

    visualization_params: Arc<Mutex<VisualizationParams>>,

    // for metaball visualization
    screen_quad_vao: WebGlVertexArrayObject,
    fb_program: WebGlProgram,
    fb_data: Option<FramebufferData>,

    frames: i32,
    start_timer: f64,
}

static GLOBAL_DATA: Lazy<Mutex<SendWrapper<Option<Rendering>>>> = Lazy::new(|| Mutex::new(SendWrapper::new(None)));

const CIRCLE_VERTEX_SHADER: &'static str = r##"#version 300 es
        uniform mat3x2 transform;

        in vec2 position;
        in vec2 uv;
        in vec3 color;

        out vec2 frag_uv;
        out vec3 frag_color;

        void main() {
            frag_uv = uv;
            frag_color = color;

            vec2 position_trans = transform * vec3(position.x, position.y, 1.);
            gl_Position = vec4(position_trans.x, position_trans.y, 0., 1.);
        }
        "##;

const CIRCLE_FRAGMENT_SHADER: &'static str = r##"#version 300 es
        precision highp float;
        out vec4 outColor;
        in vec2 frag_uv;
        in vec3 frag_color;

        // 0=circle with border, 1=metaball 
        uniform int draw_mode;
        
        void main() {
            vec2 dist2d = frag_uv - vec2(0.5, 0.5);
            float dist = 2. * sqrt(dot(dist2d, dist2d));
            if(dist > 1.) discard;

            vec3 color = frag_color;

            if(draw_mode == 0) {
                if(dist > 1. - 0.12) color = vec3(0., 0., 0.);
                outColor = vec4(color, 1.);
            } else {
                float alpha = 0.3 * (1. - dist);
                outColor = vec4(color * alpha, alpha);
            }
        }
        "##;

const LINE_VERTEX_SHADER: &'static str = r##"#version 300 es
        uniform mat3x2 transform;

        in vec2 position;
        in vec3 color;

        out vec3 frag_color;

        void main() {
            frag_color = color;

            vec2 position_trans = transform * vec3(position.x, position.y, 1.);
            gl_Position = vec4(position_trans.x, position_trans.y, 0., 1.);
        }
        "##;

const LINE_FRAGMENT_SHADER: &'static str = r##"#version 300 es
        precision highp float;
        out vec4 outColor;
        in vec3 frag_color;
        
        void main() {
            outColor = vec4(frag_color, 1.);
        }
        "##;

const FRAMEBUFFER_VERTEX_SHADER: &'static str = r##"#version 300 es
        in vec2 position;
        in vec2 uv;

        out vec2 frag_uv;

        void main() {
            frag_uv = uv;

            gl_Position = vec4(position.x, position.y, 0., 1.);
        }
        "##;

const FRAMEBUFFER_FRAGMENT_SHADER: &'static str = r##"#version 300 es
        precision highp float;

        out vec4 outColor;
        in vec2 frag_uv;

        uniform sampler2D tex;
        
        void main() {
            vec4 tcolor = texture(tex, frag_uv);

            if(tcolor.a > 0.08) {
                outColor = vec4(tcolor.rgb / tcolor.a, 1.0);
            } else {
                outColor = vec4(1., 1., 1., 1.);
            }
        }
        "##;

fn world_space_pos_and_movement_from_mouse(event: MouseEvent) -> (VF<2>, VF<2>) {
    let mut mutex_guard = GLOBAL_DATA.lock().unwrap();
    let Rendering {
        inv_transform_matrix, ..
    } = mutex_guard.as_mut().unwrap();
    world_space_pos_and_movement_general(
        event.offset_x() as FT,
        event.offset_y() as FT,
        event.movement_x() as FT,
        event.movement_y() as FT,
        *inv_transform_matrix,
    )
}

fn world_space_pos_and_movement_from_touch(event: Touch) -> (VF<2>, VF<2>) {
    let mut mutex_guard = GLOBAL_DATA.lock().unwrap();
    let Rendering {
        inv_transform_matrix, ..
    } = mutex_guard.as_mut().unwrap();
    world_space_pos_and_movement_general(
        event.page_x() as FT,
        event.page_y() as FT,
        0.,
        0.,
        *inv_transform_matrix,
    )
}

fn world_space_pos_and_movement_general(
    x: FT,
    y: FT,
    dx: FT,
    dy: FT,
    inv_transform_matrix: Matrix2x3<FT>,
) -> (VF<2>, VF<2>) {
    let window = web_sys::window().unwrap();
    let document = window.document().unwrap();
    let canvas = document.get_element_by_id("canvas").unwrap();
    let canvas: web_sys::HtmlCanvasElement = canvas.dyn_into::<web_sys::HtmlCanvasElement>().unwrap();

    let scale: f32 = window.device_pixel_ratio() as f32;

    let movement: VF<2> = inv_transform_matrix
        * Vector3::<f32>::new(
            2. * dx * scale / canvas.width() as f32,
            -(2. * dy * scale / canvas.height() as f32),
            0.,
        );
    let pos: VF<2> = inv_transform_matrix
        * Vector3::<f32>::new(
            2. * x * scale / canvas.width() as f32 - 1.,
            -(2. * y * scale / canvas.height() as f32 - 1.),
            1.,
        );

    (pos, movement)
}

pub fn init_renderer(
    input_state: Arc<Mutex<InputState>>,
    visualization_params: Arc<Mutex<VisualizationParams>>,
) -> Result<(), JsValue> {
    console_log!("start!");

    let window = web_sys::window().unwrap();
    let document = window.document().unwrap();
    let loading_message = document.get_element_by_id("loading").unwrap();
    loading_message.remove();
    let canvas = document.get_element_by_id("canvas").unwrap();
    let canvas: web_sys::HtmlCanvasElement = canvas.dyn_into::<web_sys::HtmlCanvasElement>()?;

    {
        let closure = Closure::<dyn FnMut(_)>::new(move |event: web_sys::WheelEvent| {
            let mut mutex_guard = GLOBAL_DATA.lock().unwrap();
            let Rendering { zoom, .. } = mutex_guard.as_mut().unwrap();

            *zoom *= FT::powf(10., event.delta_y() as FT * 0.001);
        });
        canvas.add_event_listener_with_callback("wheel", closure.as_ref().unchecked_ref())?;
        closure.forget();
    }
    {
        let input_state = input_state.clone();
        let closure = Closure::<dyn FnMut(_)>::new(move |event: web_sys::MouseEvent| {
            let mut input_state = input_state.lock().unwrap();
            let (pos, movement) = world_space_pos_and_movement_from_mouse(event);

            if let Some(pull_fluid_to) = &mut input_state.pull_fluid_to {
                *pull_fluid_to = pos;
            }

            let mut mutex_guard = GLOBAL_DATA.lock().unwrap();
            let Rendering {
                move_canvas,
                view_offset,
                ..
            } = mutex_guard.as_mut().unwrap();
            if *move_canvas {
                *view_offset -= movement;
            }
        });
        canvas.add_event_listener_with_callback("mousemove", closure.as_ref().unchecked_ref())?;
        closure.forget();
    }
    {
        let input_state = input_state.clone();
        let closure = Closure::<dyn FnMut(_)>::new(move |event: web_sys::MouseEvent| {
            if event.button() == 0 {
                let mut mutex_guard = GLOBAL_DATA.lock().unwrap();
                let Rendering { move_canvas, .. } = mutex_guard.as_mut().unwrap();
                *move_canvas = true;
            }
            if event.button() == 2 {
                let mut input_state = input_state.lock().unwrap();
                let (pos, _movement) = world_space_pos_and_movement_from_mouse(event);
                input_state.pull_fluid_to = Some(VF::<2>::new(pos.x, pos.y));
            }
        });
        canvas.add_event_listener_with_callback("mousedown", closure.as_ref().unchecked_ref())?;
        closure.forget();
    }

    let handle_touches = {
        let input_state = input_state.clone();
        move |touches: TouchList, touch_start: bool, touch_move: bool| {
            if touches.length() == 0 || touches.length() >= 3 {
                let mut input_state = input_state.lock().unwrap();
                input_state.pull_fluid_to = None;
            }
            if touches.length() == 1 {
                let mut input_state = input_state.lock().unwrap();
                let touch = touches.get(0).unwrap();
                let (pos, _movement) = world_space_pos_and_movement_from_touch(touch);
                input_state.pull_fluid_to = Some(VF::<2>::new(pos.x, pos.y));
            }
            if touches.length() == 2 {
                let mut input_state = input_state.lock().unwrap();
                input_state.pull_fluid_to = None;

                let touch0 = touches.get(0).unwrap();
                let touch1 = touches.get(1).unwrap();
                // let (pos0, _movement) = world_space_pos_and_movement_from_touch(touch0);
                // let (pos1, _movement) = world_space_pos_and_movement_from_touch(touch1);

                let touch_screen_pos_0 = VF::<2>::new(touch0.page_x() as FT, touch0.page_y() as FT);
                let touch_screen_pos_1 = VF::<2>::new(touch1.page_x() as FT, touch1.page_y() as FT);
                let new_touch_distance = (touch_screen_pos_1 - touch_screen_pos_0).norm();
                let new_avg_touch_pos = (touch_screen_pos_0 + touch_screen_pos_1) * 0.5;

                if touch_start {
                    let mut mutex_guard = GLOBAL_DATA.lock().unwrap();
                    let Rendering {
                        touch_distance,
                        avg_touch_pos,
                        ..
                    } = mutex_guard.as_mut().unwrap();
                    *touch_distance = new_touch_distance;
                    *avg_touch_pos = new_avg_touch_pos;
                } else if touch_move {
                    let mut mutex_guard = GLOBAL_DATA.lock().unwrap();
                    let Rendering {
                        touch_distance,
                        avg_touch_pos,
                        zoom,
                        view_offset,
                        inv_transform_matrix,
                        ..
                    } = mutex_guard.as_mut().unwrap();
                    *zoom *= *touch_distance / new_touch_distance;
                    let (_, movement) = world_space_pos_and_movement_general(
                        0.,
                        0.,
                        new_avg_touch_pos.x - avg_touch_pos.x,
                        new_avg_touch_pos.y - avg_touch_pos.y,
                        *inv_transform_matrix,
                    );
                    *view_offset -= movement;
                    *touch_distance = new_touch_distance;
                    *avg_touch_pos = new_avg_touch_pos;
                }
            }
        }
    };

    {
        let handle_touches = handle_touches.clone();
        let closure = Closure::<dyn FnMut(_)>::new(move |event: web_sys::TouchEvent| {
            let touches = event.target_touches();
            handle_touches(touches, true, false);
        });
        canvas.add_event_listener_with_callback("touchstart", closure.as_ref().unchecked_ref())?;
        closure.forget();
    }
    {
        let handle_touches = handle_touches.clone();
        let closure = Closure::<dyn FnMut(_)>::new(move |event: web_sys::TouchEvent| {
            let touches = event.target_touches();
            handle_touches(touches, false, true);
        });
        canvas.add_event_listener_with_callback("touchmove", closure.as_ref().unchecked_ref())?;
        closure.forget();
    }
    {
        let closure = Closure::<dyn FnMut(_)>::new(move |event: web_sys::TouchEvent| {
            let touches = event.target_touches();
            handle_touches(touches, false, false);
        });
        canvas.add_event_listener_with_callback("touchend", closure.as_ref().unchecked_ref())?;
        closure.forget();
    }
    {
        let input_state = input_state.clone();
        let closure = Closure::<dyn FnMut(_)>::new(move |event: web_sys::MouseEvent| {
            if event.button() == 0 {
                let mut mutex_guard = GLOBAL_DATA.lock().unwrap();
                let Rendering { move_canvas, .. } = mutex_guard.as_mut().unwrap();
                *move_canvas = false;
            }
            if event.button() == 2 {
                let mut input_state = input_state.lock().unwrap();
                input_state.pull_fluid_to = None;
            }
        });
        canvas.add_event_listener_with_callback("mouseup", closure.as_ref().unchecked_ref())?;
        closure.forget();
    }

    let context = canvas
        .get_context("webgl2")?
        .unwrap()
        .dyn_into::<WebGl2RenderingContext>()?;

    let fb_vert_shader = compile_shader(
        &context,
        WebGl2RenderingContext::VERTEX_SHADER,
        FRAMEBUFFER_VERTEX_SHADER,
    )?;
    let fb_frag_shader = compile_shader(
        &context,
        WebGl2RenderingContext::FRAGMENT_SHADER,
        FRAMEBUFFER_FRAGMENT_SHADER,
    )?;
    let fb_program = link_program(&context, &fb_vert_shader, &fb_frag_shader)?;

    let line_vert_shader = compile_shader(&context, WebGl2RenderingContext::VERTEX_SHADER, LINE_VERTEX_SHADER)?;
    let line_frag_shader = compile_shader(&context, WebGl2RenderingContext::FRAGMENT_SHADER, LINE_FRAGMENT_SHADER)?;
    let line_program = link_program(&context, &line_vert_shader, &line_frag_shader)?;
    context.use_program(Some(&line_program));

    let line_buffer = context.create_buffer().ok_or("Failed to create buffer")?;
    context.bind_buffer(WebGl2RenderingContext::ARRAY_BUFFER, Some(&line_buffer));

    let line_vao = context
        .create_vertex_array()
        .ok_or("Could not create vertex array object")?;
    context.bind_vertex_array(Some(&line_vao));

    let line_position_attr = context.get_attrib_location(&line_program, "position");
    let line_color_attr = context.get_attrib_location(&line_program, "color");

    context.vertex_attrib_pointer_with_i32(
        line_position_attr as u32,
        2,
        WebGl2RenderingContext::FLOAT,
        false,
        5 * 4,
        0 * 4,
    );
    context.vertex_attrib_pointer_with_i32(
        line_color_attr as u32 as u32,
        3,
        WebGl2RenderingContext::FLOAT,
        false,
        5 * 4,
        2 * 4,
    );
    context.enable_vertex_attrib_array(line_position_attr as u32);
    context.enable_vertex_attrib_array(line_color_attr as u32);

    context.bind_vertex_array(Some(&line_vao));

    // ///////////////////////////////////////////////////////////////////////////////////////////////
    // Circles
    // ///////////////////////////////////////////////////////////////////////////////////////////////

    let circle_vert_shader = compile_shader(&context, WebGl2RenderingContext::VERTEX_SHADER, CIRCLE_VERTEX_SHADER)?;
    let circle_frag_shader = compile_shader(
        &context,
        WebGl2RenderingContext::FRAGMENT_SHADER,
        CIRCLE_FRAGMENT_SHADER,
    )?;
    let circle_program = link_program(&context, &circle_vert_shader, &circle_frag_shader)?;
    context.use_program(Some(&circle_program));

    // let vertices: [f32; 24] = [
    //     0.0, 0.0, 0.0, 0.0,
    //     1.0, 0.0, 1.0, 0.0,
    //     0.0, 1.0, 0.0, 1.0,

    //     1.0, 1.0, 1.0, 1.0,
    //     1.0, 0.0, 1.0, 0.0,
    //     0.0, 1.0, 0.0, 1.0,
    // ];

    let circle_buffer = context.create_buffer().ok_or("Failed to create buffer")?;
    context.bind_buffer(WebGl2RenderingContext::ARRAY_BUFFER, Some(&circle_buffer));

    let circle_vao = context
        .create_vertex_array()
        .ok_or("Could not create vertex array object")?;
    context.bind_vertex_array(Some(&circle_vao));

    let uv_attribute_location = context.get_attrib_location(&circle_program, "uv");
    let position_attribute_location = context.get_attrib_location(&circle_program, "position");
    let color_attribute_location = context.get_attrib_location(&circle_program, "color");
    context.vertex_attrib_pointer_with_i32(
        position_attribute_location as u32,
        2,
        WebGl2RenderingContext::FLOAT,
        false,
        7 * 4,
        0 * 4,
    );
    context.vertex_attrib_pointer_with_i32(
        uv_attribute_location as u32,
        2,
        WebGl2RenderingContext::FLOAT,
        false,
        7 * 4,
        2 * 4,
    );
    context.vertex_attrib_pointer_with_i32(
        color_attribute_location as u32,
        3,
        WebGl2RenderingContext::FLOAT,
        false,
        7 * 4,
        4 * 4,
    );
    context.enable_vertex_attrib_array(position_attribute_location as u32);
    context.enable_vertex_attrib_array(uv_attribute_location as u32);
    context.enable_vertex_attrib_array(color_attribute_location as u32);

    let screen_quad_vao;
    {
        let screen_quad_buffer = context.create_buffer().ok_or("Failed to create buffer")?;
        context.bind_buffer(WebGl2RenderingContext::ARRAY_BUFFER, Some(&screen_quad_buffer));

        #[cfg_attr(rustfmt, rustfmt_skip)]
        let screen_quad_vertices = [
            -1., -1., 0., 0.,
            1., -1., 1., 0.,
            -1., 1., 0., 1.,

            1., -1., 1., 0.,
            -1., 1., 0., 1.,
            1., 1., 1., 1.,
        ];

        unsafe {
            let positions_array_buf_view = js_sys::Float32Array::view(&screen_quad_vertices);

            context.buffer_data_with_array_buffer_view(
                WebGl2RenderingContext::ARRAY_BUFFER,
                &positions_array_buf_view,
                WebGl2RenderingContext::STREAM_DRAW,
            );
        }

        screen_quad_vao = context
            .create_vertex_array()
            .ok_or("Could not create vertex array object")?;
        context.bind_vertex_array(Some(&screen_quad_vao));

        let uv_attribute_location = context.get_attrib_location(&fb_program, "uv");
        let position_attribute_location = context.get_attrib_location(&fb_program, "position");
        context.vertex_attrib_pointer_with_i32(
            position_attribute_location as u32,
            2,
            WebGl2RenderingContext::FLOAT,
            false,
            4 * 4,
            0 * 4,
        );
        context.vertex_attrib_pointer_with_i32(
            uv_attribute_location as u32,
            2,
            WebGl2RenderingContext::FLOAT,
            false,
            4 * 4,
            2 * 4,
        );
        context.enable_vertex_attrib_array(position_attribute_location as u32);
        context.enable_vertex_attrib_array(uv_attribute_location as u32);
    }

    // draw(&context, vert_count);

    let view_offset = VF::<2>::zeros();
    let zoom = 1.;

    // WORLD SPACE -> NORMALIZED DEVICE COORDS
    let transform_matrix: Matrix2x3<f32> = Matrix2x3::<f32>::zeros();
    let inv_transform_matrix: Matrix2x3<f32> = Matrix2x3::<f32>::zeros();

    *GLOBAL_DATA.lock().unwrap() = SendWrapper::new(Some(Rendering {
        context,

        circle_buffer,
        circle_vao,
        circle_vertices: Vec::new(),
        circle_program,

        line_buffer,
        line_vao,
        line_vertices: Vec::new(),
        line_program,

        inv_transform_matrix,
        transform_matrix,

        move_canvas: false,
        touch_distance: 0.,
        avg_touch_pos: VF::<2>::zeros(),

        view_offset,
        zoom,

        visualization_params,

        fb_program,
        fb_data: None,
        screen_quad_vao,

        frames: 0,
        start_timer: 0.,
    }));

    Ok(())
}

pub fn calc_transform_matrix(offset: VF<2>, zoom: FT, canvas_width: FT, canvas_height: FT) -> Matrix2x3<f32> {
    let div = 1.05 / FT::min(canvas_width, canvas_height);

    let min_view: V<f32, 2> =
        VF::<2>::from([-(canvas_width as f32 * div), -(canvas_height as f32 * div)]) * zoom + offset;
    let max_view: V<f32, 2> = VF::<2>::from([canvas_width as f32 * div, canvas_height as f32 * div]) * zoom + offset;

    // WORLD SPACE -> NORMALIZED DEVICE COORDS
    #[cfg_attr(rustfmt, rustfmt_skip)]
    Matrix2x3::new(
        2. / (max_view[0] - min_view[0]), 0.0, -2. * min_view[0] / (max_view[0] - min_view[0]) - 1.,
        0.0, 2. / (max_view[1] - min_view[1]), -2. * min_view[1] / (max_view[1] - min_view[1]) - 1.,
    )
}

pub fn inv_extended_coords_matrix(transform_matrix: Matrix2x3<f32>) -> Matrix2x3<f32> {
    let mut m3x3: Matrix3<f32> = transform_matrix.insert_row(2, 0.);
    m3x3[(2, 2)] = 1.0;
    assert!(m3x3.try_inverse_mut(), "Inverting matrix failed");
    m3x3.fixed_resize(0.)
}

#[wasm_bindgen]
pub fn render(time: f64) -> Result<(), JsValue> {
    struct Circle {
        pos: VF<2>,
        radius: FT,
        rgb: V<f32, 3>,
    }

    struct Line {
        from: VF<2>,
        to: VF<2>,
        rgb: [f32; 3],
        thickness: f32,
    }

    let mut points: Vec<Circle> = Vec::new();
    // points.push(Circle {
    //     pos: [1. + f32::cos(time as f32), 1. + f32::sin(time as f32)],
    //     radius: 1.,
    //     rgb: [1., 0., 0.],
    // });
    // for i in 0..10000 {
    //     points.push(Circle {
    //         pos: [
    //             0. + 0.1 * f32::cos(time as f32 + i as f32) + (i % 200) as f32 * 0.2,
    //             2. + 0.1 * f32::sin(time as f32) + (i / 200) as f32 * 0.2,
    //         ],
    //         radius: 0.1,
    //         rgb: [0., 1., 0.],
    //     });
    // }

    let mut lines: Vec<Line> = Vec::new();

    let mut rendering_mutex_guard = GLOBAL_DATA.lock().unwrap();
    let Rendering {
        context,
        circle_vertices,
        circle_buffer,
        frames,
        start_timer,
        line_vertices,
        line_program,
        line_vao,
        circle_vao,
        circle_program,
        line_buffer,
        view_offset,
        zoom,
        transform_matrix,
        inv_transform_matrix,
        visualization_params,

        fb_program,
        fb_data,
        screen_quad_vao,
        ..
    } = rendering_mutex_guard.as_mut().unwrap();

    let mut mutex_guard = super::web_loop::GLOBAL_STATE.lock().unwrap();
    let GlobalState {
        fluid_simulation,
        shared_simulation_params,
        input_state,
        ..
    } = mutex_guard.as_mut().unwrap();

    let simulation_params = { *shared_simulation_params.lock().unwrap() };
    let visualization_params = { *visualization_params.lock().unwrap() };

    type DU = DimensionUtils2d;
    const D: usize = 2;

    for i in 0..fluid_simulation.particles.position.len() {
        let pos = fluid_simulation.particles.position[i];
        let mass = fluid_simulation.particles.mass[i];

        let radius = <DU as DimensionUtils<D>>::sphere_volume_to_radius(mass / simulation_params.rest_density);
        // let color_map = get_color_map::<DU, D>(VisualizedAttribute::Velocity, simulation_params).unwrap();
        let color: Vector3<f32> = nalgebra::convert(get_color_for_particle::<DU, D>(
            i,
            &fluid_simulation.particles,
            simulation_params,
            Some(100.),
            &fluid_simulation.neighs,
            visualization_params,
        ));

        let radius_factor: f32 = match visualization_params.draw_shape {
            DrawShape::Metaball => 2.,
            _ => 1.,
        };

        points.push(Circle {
            pos,
            radius: radius * radius_factor,
            rgb: color,
        });
    }

    match &fluid_simulation.boundary_handler {
        BoundaryHandler::ParticleBasedBoundaryHandler(particle_boundary_handler) => {
            for _boundary_particle_id in 0..particle_boundary_handler.num_boundary_particles() {
                // let position = particle_boundary_handler.boundary_positions[boundary_particle_id];

                unimplemented!()
            }
        }
        BoundaryHandler::BoundaryWinchenbach2020(boundary_handler) => {
            for sdf in &boundary_handler.sdf {
                match sdf {
                    Sdf::Sdf2D(sdf2d) => {
                        for (a, b) in sdf2d.draw_lines() {
                            lines.push(Line {
                                from: a,
                                to: b,
                                rgb: [0., 0., 0.],
                                thickness: 0.005,
                            });
                        }
                    }
                    Sdf::SdfPlane(sdf_plane) => {
                        let (a, b) = sdf_plane.get_two_points_with_distance(5.);
                        lines.push(Line {
                            from: a,
                            to: b,
                            rgb: [0., 0., 0.],
                            thickness: 0.005,
                        });
                    }
                    _ => {
                        unimplemented!()
                    }
                }
            }
        }
        BoundaryHandler::NoBoundaryHandler(_) => {}
    }

    let input_state = input_state.clone();

    drop(mutex_guard);

    let time_diff = 0.2;
    if time - *start_timer > time_diff {
        console_log!("FPS: {:.2}", *frames as f64 / (time - *start_timer));
        *start_timer = time;
        *frames = 0;
    }

    *frames = *frames + 1;

    // //////////////////////////////////////////////////////////////////////////////
    // Transformations
    // //////////////////////////////////////////////////////////////////////////////

    let window = web_sys::window().unwrap();
    let document = window.document().unwrap();
    let canvas = document.get_element_by_id("canvas").unwrap();
    let canvas: web_sys::HtmlCanvasElement = canvas.dyn_into::<web_sys::HtmlCanvasElement>()?;
    let window_width = window.inner_width().unwrap().as_f64().unwrap() as u32 * 7 / 10;
    let window_height = window.inner_height().unwrap().as_f64().unwrap() as u32;
    canvas
        .style()
        .set_property("width", &format!("{}px", window_width))
        .unwrap();
    canvas
        .style()
        .set_property("height", &format!("{}px", window_height))
        .unwrap();

    let scale: f64 = window.device_pixel_ratio();

    canvas.set_width((window_width as f64 * scale).floor() as u32);
    canvas.set_height((window_height as f64 * scale).floor() as u32);
    let canvas_width = canvas.width();
    let canvas_height = canvas.height();

    context.viewport(0, 0, canvas_width as i32, canvas_height as i32);

    // WORLD SPACE -> NORMALIZED DEVICE COORDS
    *transform_matrix = calc_transform_matrix(*view_offset, *zoom, canvas_width as FT, canvas_height as FT);
    *inv_transform_matrix = inv_extended_coords_matrix(*transform_matrix);

    // let test = transform_matrix * Vector3::<f32>::new(0.5, 0.3, 1.);
    // let res = inv_transform_matrix * Vector3::<f32>::new(test[0], test[1], 1.);

    if let Some(pull_fluid_to) = input_state.lock().unwrap().pull_fluid_to {
        let animation_state = (time - (time.floor() as i64) as f64) as FT;
        let mut states = vec![
            0.,
            animation_state,
            if animation_state < 0.5 {
                animation_state + 0.5
            } else {
                animation_state - 0.5
            },
        ];
        fn smoothstep(x: FT) -> FT {
            x * x * (3. - 2. * x)
        }
        states.sort_by(|a, b| FT::partial_cmp(&a, &b).unwrap());
        for state in states {
            let smooth_state = smoothstep(state);
            let c1 = Vector3::new(0., 1., 0.);
            let c2 = Vector3::new(1., 0., 0.);
            points.push(Circle {
                pos: pull_fluid_to,
                radius: 0.05 * (1. - smooth_state),
                rgb: c1 + (c2 - c1) * (1. - (1. - state) * (1. - state)),
            });
        }

        let num_points = points.len();
        points.swap(0, num_points - 3);
        points.swap(1, num_points - 2);
        points.swap(2, num_points - 1);
    }

    // //////////////////////////////////////////////////////////////////////////////
    // Viewport
    // //////////////////////////////////////////////////////////////////////////////

    // // NORMALIZED DEVICE COORDS -> WORLD SPACE
    // #[cfg_attr(rustfmt, rustfmt_skip)]
    // let inv_det = transform_matrix[0] * transform_matrix[4] - transform_matrix[1] * transform_matrix[3];
    // #[cfg_attr(rustfmt, rustfmt_skip)]
    // let inverse_transform_matrix: [f32; 6] = [
    //     inv_det * transform_matrix[4], -inv_det * transform_matrix[1], -transform_matrix[2],
    //     -inv_det * transform_matrix[3], inv_det * transform_matrix[0], -transform_matrix[3],
    // ];

    context.clear_color(1.0, 1.0, 1.0, 1.0);
    context.clear(WebGl2RenderingContext::COLOR_BUFFER_BIT);

    // //////////////////////////////////////////////////////////////////////////////
    // Circle
    // //////////////////////////////////////////////////////////////////////////////

    circle_vertices.clear();
    for Circle { pos, radius: rad, rgb } in points {
        let (x, y, r, g, b) = (pos[0], pos[1], rgb[0], rgb[1], rgb[2]);
        #[cfg_attr(rustfmt, rustfmt_skip)]
        circle_vertices.extend_from_slice(&[
            x - rad, y - rad, 0.0, 0.0, r, g, b, // 
            x + rad, y - rad, 1.0, 0.0, r, g, b, //
            x - rad, y + rad, 0.0, 1.0, r, g, b, //
            x + rad, y + rad, 1.0, 1.0, r, g, b, //
            x + rad, y - rad, 1.0, 0.0, r, g, b, //
            x - rad, y + rad, 0.0, 1.0, r, g, b, //
        ]);
    }

    // Note that `Float32Array::view` is somewhat dangerous (hence the
    // `unsafe`!). This is creating a raw view into our module's
    // `WebAssembly.Memory` buffer, but if we allocate more pages for ourself
    // (aka do a memory allocation in Rust) it'll cause the buffer to change,
    // causing the `Float32Array` to be invalid.
    //
    // As a result, after `Float32Array::view` we have to be very careful not to
    // do any memory allocations before it's dropped.
    context.bind_buffer(WebGl2RenderingContext::ARRAY_BUFFER, Some(&circle_buffer));
    unsafe {
        let positions_array_buf_view = js_sys::Float32Array::view(&circle_vertices);

        context.buffer_data_with_array_buffer_view(
            WebGl2RenderingContext::ARRAY_BUFFER,
            &positions_array_buf_view,
            WebGl2RenderingContext::STREAM_DRAW,
        );
    }

    context.use_program(Some(&circle_program));
    context.uniform_matrix3x2fv_with_f32_array(
        context.get_uniform_location(&circle_program, "transform").as_ref(),
        false,
        transform_matrix.as_slice(),
    );

    let vert_count = (circle_vertices.len() / 7) as i32;

    match visualization_params.draw_shape {
        DrawShape::Metaball => {
            context.enable(WebGl2RenderingContext::BLEND);
            context.blend_func(WebGl2RenderingContext::ONE, WebGl2RenderingContext::ONE);

            context.uniform1i(context.get_uniform_location(&circle_program, "draw_mode").as_ref(), 1);

            let this_fb_data = fb_data
                .take()
                .map_or_else(|| FramebufferData::new(context, canvas_width, canvas_height), Ok)?;

            this_fb_data
                .ensure_size_and_bind_fb(context, canvas_width, canvas_height)
                .unwrap();

            context.clear_color(0.0, 0.0, 0.0, 0.0);
            context.clear(WebGl2RenderingContext::COLOR_BUFFER_BIT);

            context.bind_vertex_array(Some(&circle_vao));
            context.draw_arrays(WebGl2RenderingContext::TRIANGLES, 0, vert_count);

            context.disable(WebGl2RenderingContext::BLEND);

            // -----------------------------------------------------------------
            // render metaball framebuffer to display framebuffer

            context.bind_framebuffer(WebGl2RenderingContext::FRAMEBUFFER, None);
            context.viewport(0, 0, canvas_width as i32, canvas_height as i32);

            context.use_program(Some(&fb_program));

            context.active_texture(WebGl2RenderingContext::TEXTURE0);
            context.bind_texture(WebGl2RenderingContext::TEXTURE_2D, Some(&this_fb_data.tex));

            context.bind_vertex_array(Some(&screen_quad_vao));
            context.draw_arrays(WebGl2RenderingContext::TRIANGLES, 0, 6);

            *fb_data = Some(this_fb_data);
        }
        _ => {
            context.uniform1i(context.get_uniform_location(&circle_program, "draw_mode").as_ref(), 0);

            context.enable(WebGl2RenderingContext::BLEND);
            context.blend_func(
                WebGl2RenderingContext::SRC_ALPHA,
                WebGl2RenderingContext::ONE_MINUS_SRC_ALPHA,
            );

            context.bind_vertex_array(Some(&circle_vao));
            context.draw_arrays(WebGl2RenderingContext::TRIANGLES, 0, vert_count);

            context.disable(WebGl2RenderingContext::BLEND);
        }
    }

    // //////////////////////////////////////////////////////////////////////////////
    // Lines
    // //////////////////////////////////////////////////////////////////////////////

    line_vertices.clear();
    for Line {
        from,
        to,
        rgb: [r, g, b],
        thickness,
    } in lines
    {
        let dir = to - from;
        let len = dir.norm();
        let side: VF<2> = [-dir[1] / len, dir[0] / len].into();

        let p1 = from + side * thickness * 0.5;
        let p2 = from - side * thickness * 0.5;
        let p3 = to - side * thickness * 0.5;
        let p4 = to + side * thickness * 0.5;

        #[cfg_attr(rustfmt, rustfmt_skip)]
        line_vertices.extend_from_slice(&[
            p1[0], p1[1], r, g, b,
            p2[0], p2[1], r, g, b,
            p3[0], p3[1], r, g, b,

            p3[0], p3[1], r, g, b,
            p4[0], p4[1], r, g, b,
            p1[0], p1[1], r, g, b,
        ]);
    }

    context.bind_buffer(WebGl2RenderingContext::ARRAY_BUFFER, Some(&line_buffer));
    unsafe {
        let positions_array_buf_view = js_sys::Float32Array::view(&line_vertices);

        context.buffer_data_with_array_buffer_view(
            WebGl2RenderingContext::ARRAY_BUFFER,
            &positions_array_buf_view,
            WebGl2RenderingContext::STREAM_DRAW,
        );
    }

    context.use_program(Some(&line_program));
    let line_transform_uniform_location = context.get_uniform_location(&line_program, "transform");
    context.uniform_matrix3x2fv_with_f32_array(
        line_transform_uniform_location.as_ref(),
        false,
        transform_matrix.as_slice(),
    );

    context.bind_vertex_array(Some(&line_vao));

    let vert_count = (line_vertices.len() / 5) as i32;
    context.draw_arrays(WebGl2RenderingContext::TRIANGLES, 0, vert_count);

    Ok(())
}

pub fn compile_shader(context: &WebGl2RenderingContext, shader_type: u32, source: &str) -> Result<WebGlShader, String> {
    let shader = context
        .create_shader(shader_type)
        .ok_or_else(|| String::from("Unable to create shader object"))?;
    context.shader_source(&shader, source);
    context.compile_shader(&shader);

    if context
        .get_shader_parameter(&shader, WebGl2RenderingContext::COMPILE_STATUS)
        .as_bool()
        .unwrap_or(false)
    {
        Ok(shader)
    } else {
        Err(context
            .get_shader_info_log(&shader)
            .unwrap_or_else(|| String::from("Unknown error creating shader")))
    }
}

pub fn link_program(
    context: &WebGl2RenderingContext,
    vert_shader: &WebGlShader,
    frag_shader: &WebGlShader,
) -> Result<WebGlProgram, String> {
    let program = context
        .create_program()
        .ok_or_else(|| String::from("Unable to create shader object"))?;

    context.attach_shader(&program, vert_shader);
    context.attach_shader(&program, frag_shader);
    context.link_program(&program);

    if context
        .get_program_parameter(&program, WebGl2RenderingContext::LINK_STATUS)
        .as_bool()
        .unwrap_or(false)
    {
        Ok(program)
    } else {
        Err(context
            .get_program_info_log(&program)
            .unwrap_or_else(|| String::from("Unknown error creating program object")))
    }
}
