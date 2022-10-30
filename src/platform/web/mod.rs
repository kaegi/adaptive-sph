use wasm_bindgen::prelude::wasm_bindgen;

#[wasm_bindgen]
extern "C" {
    // Use `js_namespace` here to bind `console.log(..)` instead of just
    // `log(..)`
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

macro_rules! console_log {
    // Note that this is using the `log` function imported above during
    // `bare_bones`
    ($($t:tt)*) => (
        #[allow(unused_unsafe)]
        unsafe { crate::platform::web::log(&format_args!($($t)*).to_string()) }
    )
}

// pub(crate) use console_log;

mod web_loop;
mod webgl_renderer;
