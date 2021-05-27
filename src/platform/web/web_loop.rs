use std::sync::Mutex;
use std::{panic, sync::Arc};

use nalgebra::Vector2;
use once_cell::sync::Lazy;
use send_wrapper::SendWrapper;
use wasm_bindgen::{prelude::wasm_bindgen, JsValue};

use crate::floating_type_mod::FT;
use crate::{
    adaptivity::splitting::SplitPatterns, init_fluid_sim, init_simulation_params,
    properties_window::properties_window_main, simulation_parameters::SimulationParams, sph_kernels::DimensionUtils2d,
    FluidSimulation, SceneConfig,
};
use crate::{VisualizationParams, VF};

use super::webgl_renderer::init_renderer;

pub struct GlobalState {
    pub fluid_simulation: FluidSimulation<DimensionUtils2d, 2>,
    pub shared_simulation_params: Arc<Mutex<SimulationParams>>,
    pub shared_visualization_params: Arc<Mutex<VisualizationParams>>,
    pub scene_config: SceneConfig,
    pub split_patterns: SplitPatterns<2>,
    pub shared_restart: Arc<Mutex<bool>>,
    pub input_state: Arc<Mutex<InputState>>,
}

#[derive(Default, Debug, Clone, Copy)]
pub struct InputState {
    pub pull_fluid_to: Option<Vector2<FT>>,
}

pub static GLOBAL_STATE: Lazy<Mutex<SendWrapper<Option<GlobalState>>>> =
    Lazy::new(|| Mutex::new(SendWrapper::new(None)));

#[wasm_bindgen]
pub fn start() -> Result<(), JsValue> {
    panic::set_hook(Box::new(console_error_panic_hook::hook));

    let input_state = Arc::new(Mutex::new(InputState::default()));
    let shared_visualization_params = Arc::new(Mutex::new(VisualizationParams::default()));

    init_renderer(input_state.clone(), shared_visualization_params.clone())?;

    let config_yaml = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/default-config-web.yaml"));
    let scene_yaml = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/default-scene-web.yaml"));
    let split_patterns_yaml = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/split-patterns.yaml"));

    let mut simulation_params: SimulationParams =
        serde_yaml::from_str(&config_yaml).expect("failed parsing simulation config file");

    let scene_config: SceneConfig = serde_yaml::from_str(&scene_yaml).expect("failed parsing simulation config file");

    let split_patterns: SplitPatterns<2> =
        serde_yaml::from_str(&split_patterns_yaml).expect("failed parsing split patterns");

    init_simulation_params(&mut simulation_params, &scene_config);
    let fluid_simulation = init_fluid_sim(simulation_params, &scene_config, split_patterns.clone(), false);

    let shared_restart = Arc::new(Mutex::new(false));
    let shared_simulation_params = Arc::new(Mutex::new(simulation_params));

    properties_window_main(
        None,
        Arc::new(Mutex::new(false)),
        shared_restart.clone(),
        shared_simulation_params.clone(),
        shared_visualization_params.clone(),
    );

    *GLOBAL_STATE.lock().unwrap() = SendWrapper::new(Some(GlobalState {
        fluid_simulation,
        shared_simulation_params,
        split_patterns,
        scene_config,
        shared_restart,
        shared_visualization_params,
        input_state,
    }));

    Ok(())
}

#[wasm_bindgen]
pub fn simulation_step() -> Result<(), JsValue> {
    let mut mutex_guard = GLOBAL_STATE.lock().unwrap();
    let GlobalState {
        fluid_simulation,
        shared_simulation_params,
        scene_config,
        split_patterns,
        shared_restart,
        input_state,
        ..
    } = mutex_guard.as_mut().unwrap();

    let mut simulation_params = { *shared_simulation_params.lock().unwrap() };

    {
        let input_state = input_state.lock().unwrap();
        if let Some(pull_fluid_to) = input_state.pull_fluid_to {
            simulation_params.pull_fluid_to = Some(VF::<3>::from([pull_fluid_to.x, pull_fluid_to.y, 0.]));
        }
    }

    {
        let mut shared_restart_mut = shared_restart.lock().unwrap();
        if *shared_restart_mut {
            *fluid_simulation = init_fluid_sim(simulation_params, &scene_config, split_patterns.clone(), false);
            *shared_restart_mut = false;
        }
    }

    fluid_simulation.single_step(simulation_params);

    Ok(())
}
