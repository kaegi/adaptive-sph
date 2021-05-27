use std::{
    sync::{Arc, Mutex},
    thread::JoinHandle,
};

use crate::{
    floating_type_mod::FT,
    simulation_parameters::{
        BoundaryPenaltyTerm, HybridDfsphDensitySourceTerm, LevelEstimationMethod, NeighborhoodSearchAlgorithm,
        OperatorDiscretization, PressureSolverMethod, SizingFunction, SupportLengthEstimation, ViscosityType,
    },
    sph_kernels::{ParticleSizes, PARTICLE_SIZES},
    DrawShape, ETA,
};
use egui::{ScrollArea, Ui};

use crate::{
    simulation_parameters::{InitBoundaryHandlerType, SimulationParams},
    VisualizationParams, ALL_VISUALIZED_ATTIBUTES,
};

struct MyEguiApp {
    shared_cancelled: Arc<Mutex<bool>>,
    shared_restart: Arc<Mutex<bool>>,
    shared_simulation_params: Arc<Mutex<SimulationParams>>,
    shared_visualization_params: Arc<Mutex<VisualizationParams>>,

    fluid_thread: Option<JoinHandle<()>>,

    simulation_params: SimulationParams,
    visualization_params: VisualizationParams,
}

impl MyEguiApp {
    fn new(
        cc: &eframe::CreationContext<'_>,
        fluid_thread: Option<JoinHandle<()>>,
        shared_cancelled: Arc<Mutex<bool>>,
        shared_restart: Arc<Mutex<bool>>,
        shared_simulation_params: Arc<Mutex<SimulationParams>>,
        shared_visualization_params: Arc<Mutex<VisualizationParams>>,
    ) -> Self {
        // Customize egui here with cc.egui_ctx.set_fonts and cc.egui_ctx.set_visuals.
        // Restore app state using cc.storage (requires the "persistence" feature).
        // Use the cc.gl (a glow::Context) to create graphics shaders and buffers that you can use
        // for e.g. egui::PaintCallback.

        // cc.egui_ctx.set_pixels_per_point(1.2);
        cc.egui_ctx.set_visuals(egui::Visuals::light()); // Switch to light mode

        let visualization_params = { shared_visualization_params.lock().unwrap().clone() };

        let simulation_params = { shared_simulation_params.lock().unwrap().clone() };

        MyEguiApp {
            fluid_thread,
            visualization_params,
            simulation_params,
            shared_cancelled,
            shared_restart,
            shared_simulation_params,
            shared_visualization_params,
        }
    }
}

impl eframe::App for MyEguiApp {
    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        // wait for the fluid thread to finish
        *self.shared_cancelled.lock().unwrap() = true;
        self.fluid_thread.take().unwrap().join().unwrap();
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {

        egui::CentralPanel::default().show(ctx, |ui| {
            fn add_combobox<V: PartialEq>(
                ui: &mut Ui,
                label: &'static str,
                value: &mut V,
                values: Vec<(V, &'static str)>,
            ) {
                egui::ComboBox::from_label(label)
                    .selected_text(format!("{}", values.iter().find(|x| x.0 == *value).unwrap().1))
                    .show_ui(ui, |ui| {
                        for (v, label) in values {
                            ui.selectable_value(value, v, label);
                        }
                    });
            }

            // ui.heading("A");
            // ui.collapsing("Colors", |ui| {
            //     ui.horizontal(|ui| {
            //         ui.label("Your name: ");
            //         // ui.text_edit_singleline(&mut self.name);
            //     });
            // });


            ScrollArea::vertical()
                .auto_shrink([false; 2])
                .stick_to_bottom(false)
                .show_viewport(ui, |ui, _viewport| {
                    ui.set_height(1100. as f32);

                    if ui.button("Restart").clicked() {
                        *self.shared_restart.lock().unwrap() = true;
                        // self.age += 1.;
                    }

                    ui.separator();
                    

                    ui.add(egui::Slider::new(&mut self.simulation_params.cfl_factor, 0.1..=2.0).text("CFL factor"));

                    ui.add(
                        egui::Slider::from_get_set(0.1..=10., |v| {
                            if let Some(x) = v {
                                self.simulation_params.max_dt = (x / 1000.) as FT;
                            }
                            (self.simulation_params.max_dt * 1000.) as f64
                        })
                        .text("max CFL. timestep in ms"),
                    );

                    
                    ui.separator();

                    ui.add(
                        egui::Slider::new(&mut self.simulation_params.particle_radius_fine, 0.0001..=1.0)
                            .logarithmic(true)
                            .text("fine particle radius")
                            .suffix("m"),
                    );

                    ui.add(
                        egui::Slider::new(&mut self.simulation_params.particle_radius_base, 0.0001..=1.0)
                            .logarithmic(true)
                            .text("base particle radius")
                            .suffix("m"),
                    );

                    ui.add(
                        egui::Slider::new(&mut self.simulation_params.maximum_surface_distance, 0.1..=20.0)
                            .text("maximum surface distance")
                            .suffix("m"),
                    );

                    ui.separator();

                    ui.add(egui::Slider::new(&mut self.simulation_params.gravity, -20.0..=0.0).text("gravity"));

                    ui.add(
                        egui::Slider::from_get_set(1. ..=2000., |v| {
                            let scale = 100000.;
                            if let Some(x) = v {
                                self.simulation_params.viscosity = (x / scale) as FT;
                            }
                            self.simulation_params.viscosity as f64 * scale
                        })
                        .text("viscosity"),
                    );

                    // /////////////////////////////////////////////////////////////////////////////////////////////////
                    // Visualization Attributes
                    // /////////////////////////////////////////////////////////////////////////////////////////////////

                    ui.separator();

                    ui.checkbox(
                        &mut self.visualization_params.draw_support_radius,
                        "Show support radius",
                    );
                    ui.checkbox(
                        &mut self.visualization_params.show_flag_is_fluid_surface,
                        "Show fluid surface",
                    );
                    ui.checkbox(
                        &mut self.visualization_params.show_flag_neighborhood_reduced,
                        "Show neighborhood reduced",
                    );

                    egui::ComboBox::from_label("Visualized Attribute")
                        .selected_text(format!(
                            "{}",
                            self.visualization_params.visualized_attribute.as_str_lowercase()
                        ))
                        .show_ui(ui, |ui| {
                            for visualized_attribute in ALL_VISUALIZED_ATTIBUTES {
                                ui.selectable_value(
                                    &mut self.visualization_params.visualized_attribute,
                                    visualized_attribute,
                                    visualized_attribute.as_str_lowercase(),
                                );
                            }
                        });

                    add_combobox(
                        ui,
                        "Visualization Shape",
                        &mut self.visualization_params.draw_shape,
                        vec![
                            (DrawShape::FilledCircle, "filled circle"),
                            (DrawShape::FilledCircleWithBorder, "filled circle with border"),
                            (DrawShape::Circle, "circle"),
                            (DrawShape::Dot, "dot"),
                            (DrawShape::Cairo, "cairo"),
                        ],
                    );

                    ui.separator();

                    ui.checkbox(&mut self.simulation_params.check_aii, "Check a_ii");
                    ui.checkbox(&mut self.simulation_params.check_neighborhood, "Check neighborhood");
                    ui.checkbox(
                        &mut self.simulation_params.constrain_neighborhood_count,
                        "Constrain neighborhood count",
                    );
                    ui.checkbox(&mut self.simulation_params.merging, "Merge particles");
                    ui.checkbox(&mut self.simulation_params.sharing, "Share particles");
                    ui.checkbox(&mut self.simulation_params.splitting, "Split particles");

                    ui.checkbox(
                        &mut self.simulation_params.allow_merge_with_optimal_particle,
                        "Merge with optimal particles",
                    );
                    ui.checkbox(
                        &mut self.simulation_params.allow_share_with_optimal_particle,
                        "Share with optimal particles",
                    );
                    ui.checkbox(
                        &mut self.simulation_params.allow_share_with_too_small_particle,
                        "Share with too small (S) particles",
                    );
                    ui.checkbox(
                        &mut self.simulation_params.allow_merge_on_size_difference,
                        "Always allow merge to much larger particles",
                    );

                    ui.checkbox(
                        &mut self.simulation_params.boundary_is_fluid_surface,
                        "Boundary is fluid surface",
                    );

                    add_combobox(
                        ui,
                        "Level Estimation Method",
                        &mut self.simulation_params.level_estimation_method,
                        vec![
                            (LevelEstimationMethod::None, "none"),
                            (LevelEstimationMethod::CenterDiff, "center diff"),
                            (LevelEstimationMethod::EmptyAngle, "empty angle"),
                        ],
                    );

                    add_combobox(
                        ui,
                        "Neighborhood Algorithm",
                        &mut self.simulation_params.neighborhood_search_algorithm,
                        vec![
                            (NeighborhoodSearchAlgorithm::Grid, "grid"),
                            (NeighborhoodSearchAlgorithm::RStar, "r-star"),
                        ],
                    );

                    add_combobox(
                        ui,
                        "Support length estimation",
                        &mut self.simulation_params.support_length_estimation,
                        vec![
                            (SupportLengthEstimation::FromMass, "from mass"),
                            (SupportLengthEstimation::FromDistribution, "from distribution"),
                            (
                                SupportLengthEstimation::FromDistributionClamped1,
                                "from distribution (clamped 1)",
                            ),
                            (
                                SupportLengthEstimation::FromDistributionClamped2,
                                "from distribution (clamped 2)",
                            ),
                            (
                                SupportLengthEstimation::FromDistribution2,
                                "from distribution (adapted)",
                            ),
                        ],
                    );

                    // add_combobox(
                    //     ui,
                    //     "Pressure solver type",
                    //     &mut self.simulation_params.pressure_solver_method,
                    //     vec![
                    //         (PressureSolverMethod::IISPH { max_avg_density_error: 0.002 }, "from mass"),
                    //         (PressureSolverMethod::IISPH { max_avg_density_error: 0.002 }, "from distribution"),
                    //         (
                    //             SupportLengthEstimation::FromDistribution2,
                    //             "from distribution (adapted)",
                    //         ),
                    //     ],
                    // );

                    ui.horizontal(|ui| {
                        ui.label("Viscosity:");
                        ui.selectable_value(
                            &mut self.simulation_params.viscosity_type,
                            ViscosityType::WCSPH,
                            "WCSPH",
                        );
                        ui.selectable_value(
                            &mut self.simulation_params.viscosity_type,
                            ViscosityType::ApproxLaplace,
                            "Laplace",
                        );
                        ui.selectable_value(
                            &mut self.simulation_params.viscosity_type,
                            ViscosityType::ApproxLaplace,
                            "Laplace Normalized",
                        );
                    });

                    add_combobox(
                        ui,
                        "Boundary handler",
                        &mut self.simulation_params.init_boundary_handler,
                        vec![
                            (InitBoundaryHandlerType::AnalyticOverestimate, "Analytic (overestimate)"),
                            (
                                InitBoundaryHandlerType::AnalyticUnderestimate,
                                "Analytic (underestimate)",
                            ),
                            (InitBoundaryHandlerType::Particles, "Particles"),
                            (InitBoundaryHandlerType::NoBoundary, "No Boundary"),
                        ],
                    );

                    ui.separator();

                    // /////////////////////////////////////////////////////////////////////////////////////////////////
                    // Pressure solver
                    // /////////////////////////////////////////////////////////////////////////////////////////////////

                    ui.horizontal(|ui| {
                        ui.label("Pressure solver:");
                        ui.selectable_value(
                            &mut self.simulation_params.pressure_solver_method,
                            PressureSolverMethod::HybridDFSPH,
                            "Hybrid DFSPH",
                        );
                        ui.selectable_value(
                            &mut self.simulation_params.pressure_solver_method,
                            PressureSolverMethod::IISPH,
                            "IISPH",
                        );
                        ui.selectable_value(
                            &mut self.simulation_params.pressure_solver_method,
                            PressureSolverMethod::IISPH2,
                            "IISPH (with omega)",
                        );
                        ui.selectable_value(
                            &mut self.simulation_params.pressure_solver_method,
                            PressureSolverMethod::OnlyDivergence,
                            "Divergence",
                        );
                    });

                    match self.simulation_params.pressure_solver_method {
                        PressureSolverMethod::HybridDFSPH | PressureSolverMethod::OnlyDivergence => {
                            ui.add(
                                egui::Slider::from_get_set(0.01..=10., |v| {
                                    if let Some(x) = v {
                                        self.simulation_params.hybrid_dfsph_max_avg_density_error = (x / 100.) as FT;
                                    }
                                    (self.simulation_params.hybrid_dfsph_max_avg_density_error * 100.) as f64
                                })
                                .logarithmic(true)
                                .text("max. avg density deviation")
                                .suffix("%"),
                            );
                            ui.add(
                                egui::Slider::new(
                                    &mut self.simulation_params.hybrid_dfsph_max_avg_divergence_error,
                                    0.000001..=0.01,
                                )
                                .logarithmic(true)
                                .text("max. avg divergence deviation"),
                            );

                            ui.add(
                                egui::Slider::from_get_set(0.0..=5000., |v| {
                                    if let Some(x) = v {
                                        self.simulation_params.hybrid_dfsph_factor = x as FT;
                                    }
                                    self.simulation_params.hybrid_dfsph_factor as f64
                                })
                                .text("hybrid DFSPH factor in 1/s"),
                            );

                            ui.horizontal(|ui| {
                                ui.label("Density Source Term:");
                                ui.selectable_value(
                                    &mut self.simulation_params.hybrid_dfsph_density_source_term,
                                    HybridDfsphDensitySourceTerm::DensityAndDivergence,
                                    "Density & Divergence",
                                );
                                ui.selectable_value(
                                    &mut self.simulation_params.hybrid_dfsph_density_source_term,
                                    HybridDfsphDensitySourceTerm::OnlyDensity,
                                    "Only Density",
                                );
                            });
                            ui.checkbox(
                                &mut self
                                    .simulation_params
                                    .hybrid_dfsph_non_pressure_accel_before_divergence_free,
                                "'Non-pressure accels' before 'divergence free'-solver",
                            );
                        }
                        PressureSolverMethod::IISPH | PressureSolverMethod::IISPH2 => {
                            ui.add(
                                egui::Slider::from_get_set(0.01..=10., |v| {
                                    if let Some(x) = v {
                                        self.simulation_params.iisph_max_avg_density_error = (x / 100.) as FT;
                                    }
                                    (self.simulation_params.iisph_max_avg_density_error * 100.) as f64
                                })
                                .logarithmic(true)
                                .text("max. avg density deviation")
                                .suffix("%"),
                            );
                        }
                    }

                    ui.separator();

                    // /////////////////////////////////////////////////////////////////////////////////////////////////
                    // Adaptivity Parameters
                    // /////////////////////////////////////////////////////////////////////////////////////////////////


                    ui.add(
                        egui::Slider::new(&mut self.simulation_params.sdf_gradient_eps, 0.0000001..=1.)
                            .logarithmic(true)
                            .text("SDF gradient epsilon")
                            .suffix("m"),
                    );

                    ui.checkbox(
                        &mut self.simulation_params.fail_on_missing_split_pattern,
                        "Fail on missing split pattern",
                    );

                    ui.checkbox(
                        &mut self.simulation_params.use_extended_range_for_level_estimation,
                        "Use extended range for level estimation",
                    );

                    ui.checkbox(
                        &mut self.simulation_params.level_estimation_after_advection,
                        "Level estimation after advection",
                    );

                    ui.add(
                        egui::Slider::new(&mut self.simulation_params.level_estimation_range, ETA * ETA..=7.0)
                            .text("level estimation range"),
                    );

                    if self.simulation_params.init_boundary_handler == InitBoundaryHandlerType::AnalyticOverestimate
                        || self.simulation_params.init_boundary_handler
                            == InitBoundaryHandlerType::AnalyticUnderestimate
                    {
                        add_combobox(
                            ui,
                            "Boundary penalty term",
                            &mut self.simulation_params.boundary_penalty_term,
                            vec![
                                (BoundaryPenaltyTerm::None, "none"),
                                (BoundaryPenaltyTerm::Linear, "linear"),
                                (BoundaryPenaltyTerm::Quadratic1, "quadratic1"),
                                (BoundaryPenaltyTerm::Quadratic2, "quadratic2"),
                            ],
                        );
                    }

                    add_combobox(
                        ui,
                        "Sizing function",
                        &mut self.simulation_params.sizing_function,
                        vec![
                            (SizingFunction::Mass, "mass"),
                            (SizingFunction::Radius, "radius"),
                            (SizingFunction::Radius2, "radius2"),
                        ],
                    );

                    ui.add(
                        egui::Slider::new(&mut self.simulation_params.max_merge_distance, 1.0..=3.0)
                            .text("max merge distance (factor of h_ij)"),
                    );

                    ui.add(
                        egui::Slider::new(&mut self.simulation_params.max_share_distance, 1.0..=3.0)
                            .text("max share distance (factor for h_ij)"),
                    );

                    add_combobox(
                        ui,
                        "Operator discretization",
                        &mut self.simulation_params.operator_discretization,
                        vec![
                            (
                                OperatorDiscretization::ConsistentSimpleGradient,
                                "consistent (simple gradient)",
                            ),
                            (
                                OperatorDiscretization::ConsistentSymmetricGradient,
                                "consistent (symmetric gradient)",
                            ),
                            (OperatorDiscretization::Winchenbach2020, "winchenbach"),
                        ],
                    );

                    add_combobox(
                        ui,
                        "Operator discretization (M_ii)",
                        &mut self.simulation_params.operator_discretization_for_diagonal,
                        vec![
                            (None, "auto"),
                            (
                                Some(OperatorDiscretization::ConsistentSimpleGradient),
                                "consistent (simple gradient)",
                            ),
                            (
                                Some(OperatorDiscretization::ConsistentSymmetricGradient),
                                "consistent (symmetric gradient)",
                            ),
                            (Some(OperatorDiscretization::Winchenbach2020), "winchenbach"),
                        ],
                    );


                    ui.separator();

                    let mut double_precision: bool;
                    #[cfg(feature = "double-precision")]
                    {
                        double_precision = true;
                    }
                    #[cfg(not(feature = "double-precision"))]
                    {
                        double_precision = false;
                    }

                    ui.checkbox(&mut double_precision, format!("Double precision"));

                    ui.checkbox(
                        &mut (PARTICLE_SIZES == ParticleSizes::Uniform),
                        format!("Uniform particle sizes"),
                    );

                    if ui.button("Mode: Fixed particle sizes").clicked() {
                        self.simulation_params.merging = false;
                        self.simulation_params.sharing = false;
                        self.simulation_params.splitting = false;
                        self.simulation_params.pressure_solver_method = PressureSolverMethod::IISPH;
                        // self.age += 1.;
                    }

                    if ui.button("Mode: Adaptive").clicked() {
                        self.simulation_params.merging = true;
                        self.simulation_params.sharing = true;
                        self.simulation_params.splitting = true;
                        self.simulation_params.pressure_solver_method = PressureSolverMethod::HybridDFSPH;
                        // self.age += 1.;
                    }

                    {
                        *self.shared_simulation_params.lock().unwrap() = self.simulation_params;
                    }
                    {
                        *self.shared_visualization_params.lock().unwrap() = self.visualization_params.clone();
                    }

                    {
                        if *self.shared_cancelled.lock().unwrap() {
                            #[cfg(not(target_arch = "wasm32"))]
                            _frame.close();
                        }
                    }
                });
        });
    }
}

pub fn properties_window_main(
    fluid_thread: Option<JoinHandle<()>>,
    shared_cancelled: Arc<Mutex<bool>>,
    shared_restart: Arc<Mutex<bool>>,
    shared_simulation_params: Arc<Mutex<SimulationParams>>,
    shared_visualization_params: Arc<Mutex<VisualizationParams>>,
) {
    let create_app: eframe::AppCreator = Box::new(|cc| {
        Box::new(MyEguiApp::new(
            cc,
            fluid_thread,
            shared_cancelled,
            shared_restart,
            shared_simulation_params,
            shared_visualization_params,
        ))
    });

    #[cfg(not(target_arch = "wasm32"))]
    {
        let mut native_options = eframe::NativeOptions::default();
        native_options.initial_window_size = Some(egui::Vec2::new(450., 900.));
        eframe::run_native("adaptive-sph settings", native_options, create_app);
    }

    #[cfg(target_arch = "wasm32")]
    {
        let web_options = eframe::WebOptions::default();
        eframe::start_web(
            "properties_window", // hardcode it
            web_options,
            create_app,
        )
        .expect("failed to start eframe");
    }
}
