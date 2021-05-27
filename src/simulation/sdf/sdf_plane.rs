use crate::{floating_type_mod::FT, VF, vec2f};

#[derive(Clone)]
pub struct SdfPlane<const D: usize> {
    dir: VF<D>,
    delta: FT,
}

impl SdfPlane<2> {
    /**
     * Defines a 2D bounday box. The inside is empty, the infinite outside is boundary.
     */
    pub fn new_boundary_box(min: VF<2>, max: VF<2>) -> Vec<SdfPlane<2>> {
        vec![
            SdfPlane::new(vec2f(1., 0.), -min.x),
            SdfPlane::new(vec2f(-1., 0.), max.x),
            SdfPlane::new(vec2f(0., 1.), -min.y),
            SdfPlane::new(vec2f(0., -1.), max.y),
        ]
    }

    pub fn get_two_points_with_distance(&self, distance: FT) -> (VF<2>, VF<2>) {
        let line_dir = vec2f(-self.dir[1], self.dir[0]);
        (
            self.dir * self.delta + line_dir * distance / 2.,
            self.dir * self.delta - line_dir * distance / 2.,
        )
    }
}

impl<const D: usize> SdfPlane<D> {
    pub fn new(dir: VF<D>, delta: FT) -> SdfPlane<D> {
        Self { dir, delta }
    }

    pub fn probe(&self, x: VF<D>) -> FT {
        self.dir.dot(&x) + self.delta
    }

    // /** Find the nearest boundary point given a test point. Positive if outside boundary, negative if inside boundary. */
    // fn probe_with_normal(&self, x: VF<D>) -> (FT, VF<D>) {
    //     (self.probe(x), self.dir)
    // }
}
