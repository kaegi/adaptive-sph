use std::{
    fs::{create_dir_all, File},
    io::Write,
    path::PathBuf,
};

use vtkio::model::*;

use crate::{
    boundary_handler::{BoundaryHandler, BoundaryHandlerTrait},
    floating_type_mod::FT,
    sdf::Sdf,
    simulation_parameters::SimulationParams,
    sph_kernels::DimensionUtils,
    ParticleVec, V2, VF,
};

fn to_vec3d<const D: usize>(v: &VF<D>) -> [FT; 3] {
    if D == 2 {
        [v[0], v[1], 0.]
    } else {
        [v[0], v[1], v[2]]
    }
}

pub(crate) struct VtkExporter {
    /// something like './data/my-sph' which will get expanded to './data/my-sph-0001.vtk' and './data/my-sph.vtk.series'
    folder: PathBuf,
    basename: String,
    snapshot_number: usize,
    series_file: File,
}

impl VtkExporter {
    pub(crate) fn new(folder: impl Into<PathBuf>, basename: impl Into<String>) -> VtkExporter {
        let folder: PathBuf = folder.into();
        let basename: String = basename.into();

        create_dir_all(&folder).unwrap();

        let mut series_file = File::create(folder.join(format!("{}.vtk.series", basename))).unwrap();
        let series_prelude_str = "{\n\"file-series-version\": \"1.0\",\n\"files\": [";
        series_file.write_all(series_prelude_str.as_bytes()).unwrap();

        VtkExporter {
            series_file,
            folder,
            basename,
            snapshot_number: 1,
        }
    }

    pub(crate) fn add_snapshot<DU: DimensionUtils<D>, const D: usize>(
        &mut self,
        time: f32,
        particles: &ParticleVec<D>,
        boundary_handler: &BoundaryHandler<DU, D>,
        simulation_params: SimulationParams,
    ) {
        if self.snapshot_number > 1 {
            self.series_file.write_all(",".as_bytes()).unwrap();
        }

        let vtk_filename = format!("{}-{:05}.vtk", self.basename, self.snapshot_number);
        Self::write_vtk_file(
            self.folder.join(&vtk_filename),
            particles,
            boundary_handler,
            simulation_params,
        );

        write!(
            self.series_file,
            "\n{{ \"name\": \"{}\", \"time\": {} }}",
            vtk_filename, time
        )
        .unwrap();

        self.snapshot_number += 1;
    }

    fn write_vtk_file<P: Into<PathBuf>, DU: DimensionUtils<D>, const D: usize>(
        path: P,
        particles: &ParticleVec<D>,
        boundary_handler: &BoundaryHandler<DU, D>,
        simulation_params: SimulationParams,
    ) {
        // let mut color = DataArray::color_scalars("MYCOLOR", 2);
        // color.data = IOBuffer::U8(vec![0,255,128,30,10,20,30,40]);

        let mut data_ft: Vec<(String, Vec<FT>)> = Vec::new();
        let mut data_vec: Vec<(String, Vec<VF<D>>)> = Vec::new();
        let mut data_u8: Vec<(String, Vec<u8>)> = Vec::new();
        let mut lines: Vec<(VF<D>, VF<D>)> = Vec::new();

        data_ft.push(("density".into(), particles.density.clone()));
        data_ft.push(("density_error".into(), particles.density_error.clone()));
        data_ft.push(("density_error2".into(), particles.density_error2.clone()));
        data_ft.push(("pressure".into(), particles.pressure.clone()));
        data_ft.push(("mass".into(), particles.mass.clone()));
        data_ft.push(("aii".into(), particles.aii.clone()));
        data_ft.push(("h".into(), particles.h2.clone()));
        data_ft.push(("ppe_source_term".into(), particles.ppe_source_term.clone()));

        data_vec.push(("velocity".into(), particles.velocity.clone()));
        data_vec.push(("pressure_accel".into(), particles.pressure_accel.clone()));

        data_u8.push((
            "flag_is_fluid_surface".into(),
            particles.flag_is_fluid_surface.iter().map(|x| *x as u8).collect(),
        ));
        data_u8.push((
            "flag_neighborhood_reduced".into(),
            particles.flag_neighborhood_reduced.iter().map(|x| *x as u8).collect(),
        ));

        match boundary_handler {
            BoundaryHandler::BoundaryWinchenbach2020(b) => {
                for sdf in &b.sdf {
                    match sdf {
                        Sdf::Sdf2D(sdf2d) => {
                            let mut boundary_lines = sdf2d
                                .draw_lines()
                                .into_iter()
                                .map(|(s, t): (V2, V2)| {
                                    (
                                        VF::<D>::from_iterator(s.iter().cloned()),
                                        VF::<D>::from_iterator(t.iter().cloned()),
                                    )
                                })
                                .collect::<Vec<_>>();
                            lines.append(&mut boundary_lines);
                        }
                        Sdf::SdfPlane(_sdf_plane) => {
                            todo!()
                        }
                        Sdf::Sdf3D(_sdf3d) => {
                            todo!()
                        }
                    }
                }

                let distances: Vec<FT> = (0..particles.position.len())
                    .map(|i| b.distance_to_boundary(i, &particles.position, simulation_params))
                    .collect();
                data_ft.push(("distances".into(), distances));

                let lambda = (0..b.lambda.len()).map(|i| b.lambda_sum(i)).collect();

                data_ft.push(("lambda".into(), lambda));
            }
            BoundaryHandler::ParticleBasedBoundaryHandler(_b) => {
                // XXX
            }
            BoundaryHandler::NoBoundaryHandler(_b) => {}
        }

        write_vtk_file2(path, particles.position.clone(), data_ft, data_vec, data_u8, lines);

        /*let path = path.into();

        let mut vtk_density = DataArray::scalars("density", 1);
        vtk_density.data = particles.density.clone().into();

        let mut vtk_density_error = DataArray::scalars("density_error", 1);
        vtk_density_error.data = particles.density_error.clone().into();

        let mut vtk_density_error2 = DataArray::scalars("density_error2", 1);
        vtk_density_error2.data = particles.density_error2.clone().into();

        let mut vtk_pressure = DataArray::scalars("pressure", 1);
        vtk_pressure.data = particles.pressure.clone().into();

        let mut vtk_velocity = DataArray::scalars("velocity", 3);
        vtk_velocity.data = particles.velocity.iter().flat_map(to_vec3d).collect::<Vec<_>>().into();

        let mut vtk_mass = DataArray::scalars("mass", 1);
        vtk_mass.data = particles.mass.clone().into();

        let mut vtk_flag1 = DataArray::scalars("flag1", 1);
        vtk_flag1.data = particles.flag1.iter().cloned().map(u8::from).collect::<Vec<_>>().into();

        let mut vtk_flag2 = DataArray::scalars("flag2", 1);
        vtk_flag2.data = particles.flag2.iter().cloned().map(u8::from).collect::<Vec<_>>().into();

        let mut vtk_pressure_accel = DataArray::scalars("pressure_accel", 3);
        vtk_pressure_accel.data = particles
            .pressure_accel
            .iter()
            .flat_map(to_vec3d)
            .collect::<Vec<_>>()
            .into();

        let mut vtk_aii = DataArray::scalars("aii", 1);
        vtk_aii.data = particles.aii.clone().into();

        let num_particles = particles.position.len();
        let vtk_points: Vec<FT> = particles.position.iter().flat_map(to_vec3d).collect();
        let vtk_verts: Vec<u32> = (0..num_particles).flat_map(|i| [1, i as u32]).collect();

        let vtk = Vtk {
            version: Version::new((4, 2)),
            byte_order: ByteOrder::BigEndian,
            title: String::from("SPH Particles 1.0"),
            file_path: Some(path.clone()),
            data: DataSet::inline(PolyDataPiece {
                points: vtk_points.into(),
                verts: VertexNumbers::Legacy {
                    num_cells: num_particles as u32,
                    vertices: vtk_verts,
                }
                .into(),
                // cells: Cells {
                //     cell_verts: VertexNumbers::Legacy {
                //         num_cells: 4,
                //         vertices: vec![1, 0, 1, 1, 1, 2, 1, 3],
                //     },
                //     types: vec![CellType::Vertex, CellType::Vertex, CellType::Vertex, CellType::Vertex],
                // },
                data: Attributes {
                    cell: Vec::new(),
                    point: vec![
                        Attribute::DataArray(vtk_pressure),
                        Attribute::DataArray(vtk_density),
                        Attribute::DataArray(vtk_density_error),
                        Attribute::DataArray(vtk_density_error2),
                        Attribute::DataArray(vtk_mass),
                        Attribute::DataArray(vtk_velocity),
                        Attribute::DataArray(vtk_flag1),
                        Attribute::DataArray(vtk_flag2),
                        Attribute::DataArray(vtk_pressure_accel),
                        Attribute::DataArray(vtk_aii),
                    ],
                },
                ..Default::default()
            }),
        };
        vtk.export(path).unwrap();
        */

        // let vtk_xml: VTKFile = vtk.try_into_xml_format(Compressor::LZMA, 9).unwrap();
        // let f = std::fs::File::create("sph_lzma.pvd").unwrap();
        // let writer = BufWriter::new(f);
        // let mut serializer = quick_xml::se::Serializer::new(writer);
        // vtk_xml.serialize(&mut serializer).unwrap();
    }
}

impl Drop for VtkExporter {
    fn drop(&mut self) {
        let series_end_str = "\n]\n}";
        self.series_file.write_all(series_end_str.as_bytes()).unwrap();
    }
}

pub fn write_vtk_file2<P: Into<PathBuf>, const D: usize>(
    path: P,
    mut positions: Vec<VF<D>>,
    data_ft: Vec<(String, Vec<FT>)>,
    data_vec: Vec<(String, Vec<VF<D>>)>,
    data_u8: Vec<(String, Vec<u8>)>,
    lines: Vec<(VF<D>, VF<D>)>,
) {
    // let mut color = DataArray::color_scalars("MYCOLOR", 2);
    // color.data = IOBuffer::U8(vec![0,255,128,30,10,20,30,40]);

    let mut data_arrays: Vec<DataArray> = Vec::new();
    let num_lines = lines.len();

    for (name, mut arr) in data_ft {
        let mut data_array = DataArray::scalars(name, 1);

        for _ in 0..2 * num_lines {
            // add dummy data for line cells
            arr.push(0.);
        }

        data_array.data = arr.into();
        data_arrays.push(data_array);
    }

    for (name, mut arr) in data_vec {
        let mut data_array = DataArray::scalars(name, 3);

        for _ in 0..2 * num_lines {
            // add dummy data for line cells
            arr.push(VF::<D>::zeros());
        }

        data_array.data = arr.iter().flat_map(to_vec3d).collect::<Vec<_>>().into();
        data_arrays.push(data_array);
    }

    for (name, mut arr) in data_u8 {
        let mut data_array = DataArray::scalars(name, 1);

        for _ in 0..2 * num_lines {
            // add dummy data for line cells
            arr.push(0);
        }

        data_array.data = arr.into();
        data_arrays.push(data_array);
    }

    let path = path.into();

    let num_particles = positions.len();
    let vtk_verts: Vec<u32> = (0..num_particles).flat_map(|i| [1, i as u32]).collect();

    let mut vtk_line_indices: Vec<u32> = Vec::new();
    for (a, b) in lines {
        let a_idx = positions.len();
        positions.push(a);

        let b_idx = positions.len();
        positions.push(b);

        vtk_line_indices.push(2);
        vtk_line_indices.push(a_idx as u32);
        vtk_line_indices.push(b_idx as u32);
    }

    let vtk_points: Vec<FT> = positions.iter().flat_map(|x: &VF<D>| to_vec3d(x)).collect();

    let vtk = Vtk {
        version: Version::new((4, 2)),
        byte_order: ByteOrder::BigEndian,
        title: String::from("SPH Particles 1.0"),
        file_path: Some(path.clone()),
        data: DataSet::PolyData {
            meta: None,
            pieces: vec![Piece::Inline(Box::new(PolyDataPiece {
                points: vtk_points.into(),
                verts: VertexNumbers::Legacy {
                    num_cells: num_particles as u32,
                    vertices: vtk_verts,
                }
                .into(),
                lines: VertexNumbers::Legacy {
                    num_cells: num_lines as u32,
                    vertices: vtk_line_indices,
                }
                .into(),
                // cells: Cells {
                //     cell_verts: VertexNumbers::Legacy {
                //         num_cells: 4,
                //         vertices: vec![1, 0, 1, 1, 1, 2, 1, 3],
                //     },
                //     types: vec![CellType::Vertex, CellType::Vertex, CellType::Vertex, CellType::Vertex],
                // },
                data: Attributes {
                    cell: Vec::new(),
                    point: data_arrays.into_iter().map(Attribute::DataArray).collect(),
                },
                ..Default::default()
            }))],
        },
    };
    vtk.export(path).unwrap();

    // let vtk_xml: VTKFile = vtk.try_into_xml_format(Compressor::LZMA, 9).unwrap();
    // let f = std::fs::File::create("sph_lzma.pvd").unwrap();
    // let writer = BufWriter::new(f);
    // let mut serializer = quick_xml::se::Serializer::new(writer);
    // vtk_xml.serialize(&mut serializer).unwrap();
}
