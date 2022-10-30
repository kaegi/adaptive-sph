use crate::{floating_type_mod::FT, vec2f, V2, VF};

#[derive(Clone)]
pub struct Sdf2DConnectedComponents {
    /**
     * These points describe a 2D object which is defined by the lines between points (0, 1), (1, 2), ..., (n-1, n) and line (n, 0).
     * The left hand side of these lines (when going from line start to line end) points to the inside of the scene (air), the right side points to outside of the scene (solid).
     */
    pub point: Vec<VF<2>>,

    /// The direction of the line between the adjacent lines (i-1, i) and (i, i+1) that points inwards (into air).
    point_pseudo_normal: Vec<VF<2>>,

    normalized_line_dir: Vec<VF<2>>,
}

enum ClosestObject {
    Point {
        point_idx: usize,
        dist_sq: FT,
        point_dir: VF<2>,
    },

    // start index of the line
    Line {
        line_start_idx: usize,
        dist: FT,
        left_normalized_dir: VF<2>,
    },
}

fn rotate_left_90_degrees(v: VF<2>) -> VF<2> {
    vec2f(-v.y, v.x)
}

impl Sdf2DConnectedComponents {
    fn from_points(points: &[VF<2>]) -> Sdf2DConnectedComponents {
        let mut normalized_line_dir: Vec<VF<2>> = Vec::with_capacity(points.len());
        let mut point_pseudo_normal: Vec<VF<2>> = Vec::with_capacity(points.len());

        for i in 0..points.len() {
            let mut line_dir = points[(i + 1) % points.len()] - points[i];
            assert!(line_dir.norm_squared() > 0.00001);
            line_dir.normalize_mut();
            normalized_line_dir.push(line_dir);
        }

        for i in 0..points.len() {
            let prev_line_dir = normalized_line_dir[if i == 0 { points.len() - 1 } else { i - 1 }];
            let next_line_dir = normalized_line_dir[i];

            let prev_line_left = rotate_left_90_degrees(prev_line_dir);
            let next_line_left = rotate_left_90_degrees(next_line_dir);

            let pseudo_normal = prev_line_left + next_line_left;

            // This check is needed for numerical stability. "pseudoNormal = 0" means "prevLineLeft = -nextLineLeft"
            // which means that the lines go in opposite directions directly over each other -> zero width -> non-manifold mesh.
            assert!(pseudo_normal.norm_squared() > 0.00001);

            point_pseudo_normal.push(pseudo_normal);
        }

        Sdf2DConnectedComponents {
            point: points.to_vec(),
            normalized_line_dir,
            point_pseudo_normal,
        }
    }

    fn find_min_dist_object(&self, x: VF<2>) -> (ClosestObject, FT) {
        let mut min_dist_object = ClosestObject::Point {
            point_idx: 0,
            dist_sq: FT::INFINITY,
            point_dir: VF::<2>::zeros(),
        };
        let mut min_dist_sq = FT::INFINITY;

        for line_start_idx in 0..self.point.len() {
            let line_start = self.point[line_start_idx];
            let line_end = self.point[(line_start_idx + 1) % self.point.len()];
            let line_len_sq = (line_end - line_start).norm_squared();
            let line_dir = self.normalized_line_dir[line_start_idx];
            let point_dir = x - line_start;
            let left_normalized_dir: VF<2> = rotate_left_90_degrees(line_dir);
            // let rightDir = -leftDir;

            let projection_len = point_dir.dot(&line_dir);
            if projection_len > 0. && projection_len * projection_len < line_len_sq {
                let point_line_dist = point_dir.dot(&left_normalized_dir);
                let point_line_dist_sq = point_line_dist * point_line_dist;
                if point_line_dist_sq < min_dist_sq {
                    min_dist_object = ClosestObject::Line {
                        line_start_idx,
                        dist: point_line_dist,
                        left_normalized_dir,
                    };
                    min_dist_sq = point_line_dist_sq;
                }
            }

            // ---------------------------
            // corner analysis
            let corner_dist_sq = point_dir.norm_squared();
            if corner_dist_sq < min_dist_sq {
                min_dist_object = ClosestObject::Point {
                    point_idx: line_start_idx,
                    dist_sq: corner_dist_sq,
                    point_dir,
                };
                min_dist_sq = corner_dist_sq;
            }
        }

        assert!(min_dist_sq != FT::INFINITY);

        (min_dist_object, min_dist_sq)
    }

    fn to_dist_and_dir(&self, min_dist_object: ClosestObject) -> (FT, VF<2>) {
        match min_dist_object {
            ClosestObject::Point {
                point_idx,
                dist_sq,
                point_dir,
            } => {
                let sign = if self.point_pseudo_normal[point_idx].dot(&point_dir) >= 0. {
                    1.0
                } else {
                    -1.0
                };
                let dist = dist_sq.sqrt();
                (dist * sign, sign * -point_dir / dist)
            }
            ClosestObject::Line {
                line_start_idx: _line_start_idx,
                dist,
                left_normalized_dir,
            } => (dist, -left_normalized_dir),
        }
    }
}

#[derive(Clone)]
pub struct Sdf2D {
    connected_components: Vec<Sdf2DConnectedComponents>,
}

impl Sdf2D {
    /**
     * Defines a 2D bounday box. The inside is empty, the infinite outside is boundary.
     */
    pub fn new_boundary_box(min: VF<2>, max: VF<2>) -> Sdf2D {
        let mut connected_components = Vec::new();

        connected_components.push(Sdf2DConnectedComponents::from_points(&[
            vec2f(min.x, min.y),
            vec2f(max.x, min.y),
            vec2f(max.x, max.y),
            vec2f(min.x, max.y),
        ]));

        Sdf2D { connected_components }
    }

    /** This function is meant for drawing/debug output. */
    pub fn draw_lines(&self) -> Vec<(V2, V2)> {
        let mut result: Vec<(V2, V2)> = Vec::new();

        for connected_component in &self.connected_components {
            for i in 0..connected_component.point.len() {
                result.push((
                    connected_component.point[i],
                    connected_component.point[(i + 1) % connected_component.point.len()],
                ));
            }
        }

        result
    }

    pub fn probe(&self, x: VF<2>) -> FT {
        self.probe_with_normal(x).0
    }

    /** Find the nearest boundary point given a test point. Positive if outside boundary, negative if inside boundary. */
    fn probe_with_normal(&self, x: VF<2>) -> (FT, VF<2>) {
        let mut min_dist_sq = FT::INFINITY;
        let mut min_dist_object = ClosestObject::Point {
            point_idx: 0,
            dist_sq: FT::INFINITY,
            point_dir: VF::<2>::zeros(),
        };
        let mut min_dist_component = &self.connected_components[0];

        for connected_component in &self.connected_components {
            let (min_dist_component_object, dist_sq) = connected_component.find_min_dist_object(x);

            if dist_sq < min_dist_sq {
                min_dist_sq = dist_sq;
                min_dist_object = min_dist_component_object;
                min_dist_component = connected_component;
            }
        }

        assert!(min_dist_sq.is_finite());

        min_dist_component.to_dist_and_dir(min_dist_object)
    }
}
