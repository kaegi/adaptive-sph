
use rstar::{primitives::GeomWithData, Point, RTree};

const MAX_NEIGHBOR_COUNT: usize = 20000;

use crate::{
    concurrency::par_iter_mut1,
    simulation_parameters::{NeighborhoodSearchAlgorithm, SimulationParams},
    sph_kernels::{smoothing_length, smoothing_length_single, DimensionUtils, ParticleSizes, PARTICLE_SIZES},
    floating_type_mod::FT, V, VF, VI,
};

pub struct NeighborhoodCache {
    neighs: Vec<Vec<u32>>,
}

impl NeighborhoodCache {
    pub fn new(num_particles: usize) -> Self {
        NeighborhoodCache {
            neighs: (0..num_particles).map(|_| Vec::new()).collect(),
        }
    }

    pub fn internal_lists(&self) -> &Vec<Vec<u32>> {
        &self.neighs
    }

    pub fn iter<'a>(&'a self, i: usize) -> impl Iterator<Item = usize> + 'a {
        self.neighs[i].iter().map(|&x| x as usize)
    }

    pub fn neighbor_count(&self, i: usize) -> usize {
        self.neighs[i].len()
    }

    pub fn len(&self) -> usize {
        self.neighs.len()
    }

    pub fn swap(&mut self, i: usize, j: usize) {
        self.neighs.swap(i, j);
    }

    pub fn truncate(&mut self, len: usize) {
        self.neighs.truncate(len);
    }

    pub fn extend(&mut self, num_elements: usize) {
        self.neighs.extend((0..num_elements).map(|_| Vec::new()));
    }

    pub fn pop(&mut self) {
        self.neighs.pop();
    }

    pub fn filter_down<DU: DimensionUtils<D>, const D: usize>(
        &mut self,
        simulation_params: SimulationParams,
        h: &[FT],
        position: &[VF<D>],
        support_length_by_smoothing_length: FT,
    ) {
        par_iter_mut1(&mut self.neighs, |i, neigh_list| {
            neigh_list.retain(|&j| {
                let x_ij_sq = (position[i] - position[j as usize]).norm_squared();
                let s_ij = smoothing_length(h, i, j as usize, simulation_params) * support_length_by_smoothing_length;
                x_ij_sq < s_ij * s_ij
            });
        });
    }

    #[inline(always)]
    pub fn build_neighborhood_list_rstar<DU: DimensionUtils<D>, const D: usize>(
        &mut self,
        simulation_params: SimulationParams,

        // these are entries for every fluid particle
        fluid_particle_positions: &[VF<D>],
        h: &[FT],
        support_length_by_smoothing_length: FT,
    ) {
        #[derive(Debug, PartialEq, Clone, Copy)]
        struct CustomRTreePoint<const D: usize> {
            p: VF<D>,
        }
        impl<const D: usize> Point for CustomRTreePoint<D> {
            type Scalar = FT;

            const DIMENSIONS: usize = D;

            fn generate(mut generator: impl FnMut(usize) -> Self::Scalar) -> Self {
                CustomRTreePoint {
                    p: VF::<D>::from_iterator((0..D).map(|d| generator(d))),
                }
            }

            fn nth(&self, index: usize) -> Self::Scalar {
                self.p[index]
            }

            fn nth_mut(&mut self, index: usize) -> &mut Self::Scalar {
                &mut self.p[index]
            }
        }
        impl<const D: usize> From<VF<D>> for CustomRTreePoint<D> {
            fn from(p: VF<D>) -> Self {
                CustomRTreePoint { p }
            }
        }

        type CustomRTreeElem<const D: usize> = GeomWithData<CustomRTreePoint<D>, usize>;

        let rtree_elems: Vec<_> = fluid_particle_positions
            .iter()
            .enumerate()
            .map(|(idx, neigh_pos)| CustomRTreeElem::new(CustomRTreePoint::from(*neigh_pos), idx))
            .collect();

        let rtree = RTree::<CustomRTreeElem<D>>::bulk_load(rtree_elems);


        let num_fluid_particles = fluid_particle_positions.len();


        par_iter_mut1(&mut self.neighs, |i, p_neighs| {
            p_neighs.clear();

            let mut num_neighbors = 0;

            let this_particle_position = fluid_particle_positions[i];

            let support_radius_i =
                smoothing_length_single(h, i, simulation_params) * support_length_by_smoothing_length;
            let max_dist_sq = support_radius_i * support_radius_i;

            for neigh_point in rtree.locate_within_distance(CustomRTreePoint::from(this_particle_position), max_dist_sq)
            {
                let j = neigh_point.data;

                if PARTICLE_SIZES == ParticleSizes::Adaptive {
                    // skip all neighbors that are larger than the symmetrized support radius
                    let neigh_particle_position = fluid_particle_positions[j];
                    let x_ij_sq = (this_particle_position - neigh_particle_position).norm_squared();
                    assert!(x_ij_sq <= max_dist_sq);
                    let s_ij = smoothing_length(h, i, j, simulation_params) * support_length_by_smoothing_length;
                    if x_ij_sq >= s_ij * s_ij {
                        continue;
                    }
                }

                if num_neighbors == MAX_NEIGHBOR_COUNT {
                    panic!("exceeded maximum allowed number of {} neighbors", MAX_NEIGHBOR_COUNT);
                }
                p_neighs.push(j as u32);
                num_neighbors = num_neighbors + 1;
            }
        });

        if PARTICLE_SIZES == ParticleSizes::Adaptive {
            // symetrize neighbor lists (every particle only found smaller neighbors so far)
            let mut new_neighborhood_list_indices = self.neighs.clone();
            for i in 0..num_fluid_particles {
                let xi = fluid_particle_positions[i];

                for j in self.iter(i) {
                    if j == i {
                        continue;
                    }

                    // h[j] * h[j]

                    // if h[j] >= h[i] {
                    //     unreachable!("every particle is supposed to only neighbors with smaller support radius: h[i={}]={} h[j={}]={}", i, h[i], j, h[j]);
                    // }

                    let xj = fluid_particle_positions[j];
                    let x_ij_sq = (xi - xj).norm_squared();
                    let sr_ij = smoothing_length(h, i, j, simulation_params) * support_length_by_smoothing_length;
                    assert!(x_ij_sq < sr_ij * sr_ij);
                    let sr_j = smoothing_length_single(h, j, simulation_params) * support_length_by_smoothing_length;
                    if x_ij_sq > sr_j * sr_j {
                        new_neighborhood_list_indices[j].push(i as u32);
                    }
                }
            }

            self.neighs = new_neighborhood_list_indices;

            if simulation_params.check_neighborhood {
                println!("=====> SLOW: CHECK R-STAR NEIGHBORHOOD <=====");

                for i in 0..num_fluid_particles {
                    // check: I is neighbor of I
                    assert!(
                        self.iter(i).any(|j| j == i),
                        "r-star neighbor search bug: particle is not neighbor of itself"
                    );

                    // check: J is neighbor of I => I is neighbor of J
                    assert!(
                        self.iter(i).all(|j| self.iter(j).any(|i2| i == i2)),
                        "r-star neighbor search bug: neighbor is not reflexive"
                    );

                    // check: there are no duplicate neighbors
                    let mut neighs_i: Vec<usize> = self.iter(i).collect();
                    let neighs_i_len = neighs_i.len();
                    neighs_i.sort();
                    neighs_i.dedup();
                    assert!(
                        neighs_i_len == neighs_i.len(),
                        "r-star neighbor search bug: duplicate entries"
                    );
                }

                // check: we find a neighbor IFF x_ij < (h[i] + h[j]) * 0.5
                for i in 0..num_fluid_particles {
                    for j in 0..num_fluid_particles {
                        let xi = fluid_particle_positions[i];
                        let xj = fluid_particle_positions[j];
                        let x_ij_sq = (xi - xj).norm_squared();
                        let sr_ij = smoothing_length(h, i, j, simulation_params) * support_length_by_smoothing_length;

                        let interact = x_ij_sq < sr_ij * sr_ij;
                        let i_has_neighbor_j = self.iter(i).any(|k| k == j);
                        let j_has_neighbor_i = self.iter(j).any(|k| k == i);

                        assert_eq!(i_has_neighbor_j, interact);
                        if j_has_neighbor_i != interact {
                            panic!(
                                "j_has_neighbor_i:{} != interact:{}  x_ij_sq:{}  sr_ij_sq:{}",
                                j_has_neighbor_i,
                                interact,
                                x_ij_sq,
                                sr_ij * sr_ij
                            );
                        }
                    }
                }
            }
        }

    }

    #[inline(always)]
    pub fn build_neighborhood_list_grid<DU: DimensionUtils<D>, const D: usize>(
        &mut self,

        // this can be the position of all fluid particles or all boundary particles
        neighbor_positions: &[VF<D>],

        // these are entries for every fluid particle
        fluid_particle_positions: &[VF<D>],
        support_radius: FT,
    ) {
        fn particle_to_cell_pos<const D: usize>(particle_pos: VF<D>, kernel_support_radius: FT) -> VI<D> {
            (particle_pos / kernel_support_radius).map(|x| x.floor() as i32)
        }

        if neighbor_positions.len() == 0 {
            for p_neighborhood_list_indices in &mut self.neighs {
                p_neighborhood_list_indices.clear();
            }
            return;
        }

        let mut domain_min = neighbor_positions[0];
        let mut domain_max = neighbor_positions[0];
        for position in neighbor_positions {
            for d in 0..D {
                domain_min[d] = FT::min(domain_min[d], position[d]);
                domain_max[d] = FT::max(domain_max[d], position[d]);
            }
        }

        let cells_min = domain_min.map(|x| (x / support_radius).floor() as i32 - 1);
        let cells_max = domain_max.map(|x| (x / support_radius).floor() as i32 + 2);
        let grid_size: V<usize, D> = (cells_max - cells_min).map(|x| x as usize);

        let mut grid: CellGrid<D> = CellGrid::new(cells_min, grid_size);

        for (particle_id, position) in neighbor_positions.iter().enumerate() {
            let cell_pos = particle_to_cell_pos(*position, support_radius);
            grid.get_mut(cell_pos).particle_ids.push(particle_id);
        }

        par_iter_mut1(&mut self.neighs, |particle_id, p_neighs| {
            p_neighs.clear();

            let mut num_neighbors = 0;

            let this_particle_position = fluid_particle_positions[particle_id];

            let particle_cell_pos = particle_to_cell_pos(this_particle_position, support_radius);

            DU::iterate_grid_neighbors(1, |offset| {
                let cell_pos = particle_cell_pos + offset;

                for d in 0..D {
                    if cell_pos[d] < cells_min[d] || cell_pos[d] >= cells_max[d] {
                        return;
                    }
                }

                for &neigh_particle_id in &grid.get(cell_pos).particle_ids {
                    // if neigh_particle_id == particle_id { continue; }

                    let neigh_particle_position = neighbor_positions[neigh_particle_id];

                    if (neigh_particle_position - this_particle_position).norm_squared()
                        >= support_radius * support_radius
                    {
                        continue;
                    }

                    if num_neighbors == MAX_NEIGHBOR_COUNT {
                        panic!("exceeded maximum allowed number of {} neighbors", MAX_NEIGHBOR_COUNT);
                    }
                    p_neighs.push(neigh_particle_id as u32);
                    num_neighbors = num_neighbors + 1;
                }
            });
        });
    }
}

#[inline(always)]
pub fn build_neighborhood_list<DU: DimensionUtils<D>, const D: usize>(
    simulation_params: SimulationParams,

    // these are entries for every fluid particle
    fluid_particle_positions: &[VF<D>],
    fluid_h: &[FT],
    support_length_by_smoothing_length: FT,
    neighs: &mut NeighborhoodCache,
) {
    match simulation_params.neighborhood_search_algorithm {
        NeighborhoodSearchAlgorithm::Grid => {
            assert!(PARTICLE_SIZES == ParticleSizes::Uniform);
            neighs.build_neighborhood_list_grid::<DU, D>(
                fluid_particle_positions,
                fluid_particle_positions,
                simulation_params.h * DU::support_radius_by_smoothing_length(),
            );
        }
        NeighborhoodSearchAlgorithm::RStar => {
            // assert!(PARTICLE_SIZES != ParticleSizes::Uniform);
            neighs.build_neighborhood_list_rstar::<DU, D>(
                simulation_params,
                fluid_particle_positions,
                fluid_h,
                support_length_by_smoothing_length,
            );
        }
    }

}

struct Cell {
    particle_ids: Vec<usize>,
}

impl Cell {
    fn new() -> Cell {
        Cell {
            particle_ids: Vec::new(),
        }
    }
}

struct CellGrid<const D: usize> {
    grid_min: V<i32, D>,
    size: V<usize, D>,
    cells: Vec<Cell>,
}

impl<const D: usize> CellGrid<D> {
    fn new(grid_min: V<i32, D>, grid_size: V<usize, D>) -> CellGrid<D> {
        let num_elements = grid_size.fold(1, |acc, x| acc * x);
        CellGrid {
            grid_min,
            size: grid_size,
            cells: (0..num_elements).map(|_| Cell::new()).collect(),
        }
    }

    fn pos_to_idx(&self, mut cell_pos: V<i32, D>) -> usize {
        cell_pos = cell_pos - self.grid_min;

        let mut multiplier = 1;
        let mut idx: usize = 0;
        for d in 0..D {
            assert!(0 <= cell_pos[d]);
            assert!((cell_pos[d] as usize) < self.size[d]);
            idx += multiplier * cell_pos[d] as usize;
            multiplier *= self.size[d];
        }
        idx
    }

    fn get(&self, cell_pos: V<i32, D>) -> &Cell {
        let idx = self.pos_to_idx(cell_pos);
        self.cells
            .get(idx)
            .expect("out-of-bounds access should have been catched before")
    }

    fn get_mut(&mut self, cell_pos: V<i32, D>) -> &mut Cell {
        let idx = self.pos_to_idx(cell_pos);
        self.cells
            .get_mut(idx)
            .expect("out-of-bounds access should have been catched before")
    }
}
