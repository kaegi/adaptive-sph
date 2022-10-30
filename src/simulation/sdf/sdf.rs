use crate::{floating_type_mod::FT, vec2f, vec3f, VF};

use super::{Sdf2D, Sdf3D, SdfPlane};

#[derive(Clone)]
pub enum Sdf<const D: usize> {
    Sdf2D(Sdf2D),
    Sdf3D(Sdf3D),
    SdfPlane(SdfPlane<D>),
}

impl From<Sdf2D> for Sdf<2> {
    fn from(v: Sdf2D) -> Sdf<2> {
        Sdf::<2>::Sdf2D(v)
    }
}

impl From<SdfPlane<2>> for Sdf<2> {
    fn from(v: SdfPlane<2>) -> Sdf<2> {
        Sdf::<2>::SdfPlane(v)
    }
}

impl<const D: usize> Sdf<D> {
    pub fn probe(&self, x: VF<D>) -> FT {
        if D == 2 {
            let x2: VF<2> = vec2f(x[0], x[1]);

            match self {
                Sdf::Sdf2D(sdf) => sdf.probe(x2),
                Sdf::Sdf3D(_) => unreachable!(),
                Sdf::SdfPlane(sdf) => sdf.probe(x),
            }
        } else if D == 3 {
            let x3: VF<3> = vec3f(x[0], x[1], x[2]);

            match self {
                Sdf::Sdf2D(_) => unreachable!(),
                Sdf::Sdf3D(sdf) => sdf.probe(x3),
                Sdf::SdfPlane(sdf) => sdf.probe(x),
            }
        } else {
            unreachable!()
        }
    }

    /**
     * This gradient is NOT normalized.
     */
    pub fn finite_diff_gradient(&self, x: VF<D>, eps: FT) -> VF<D> {
        let inv_2eps = 1. / (2. * eps);

        let iter = (0..D).map(|i| {
            let mut xp = x;
            let mut xn = x;
            xp[i] += eps;
            xn[i] -= eps;
            (self.probe(xp) - self.probe(xn)) * inv_2eps
        });

        VF::<D>::from_iterator(iter)
    }
}
