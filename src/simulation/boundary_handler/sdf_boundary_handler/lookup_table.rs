use crate::floating_type_mod::FT;

pub struct LookupTable1D {
    min: FT,
    max: FT,
    len_inv: FT,
    steps: usize,
    data: Vec<FT>,
}

impl LookupTable1D {
    pub fn new(min: FT, max: FT, steps: usize, f: impl Fn(FT) -> FT) -> Self {
        assert!(steps > 1);

        let mut data = Vec::with_capacity(steps + 1);
        for i in 0..steps + 1 {
            let x = (i as FT / steps as FT) * (max - min) + min;
            let y = f(x);
            assert!(y.is_finite());
            data.push(y);
        }

        Self {
            min,
            max,
            len_inv: 1. / (max - min),
            steps,
            data,
        }
    }

    pub fn get(&self, x: FT) -> FT {
        assert!(x >= self.min);
        assert!(x < self.max);

        let fidx = (x - self.min) * self.len_inv * self.steps as FT;

        assert!(fidx >= 0.);

        let fidx_floor = fidx.floor();
        let interp = fidx - fidx_floor;
        let idx = fidx_floor as usize;
        if idx + 1 >= self.data.len() {
            return self.data[idx];
        } else {
            self.data[idx] * (1. - interp) + self.data[idx + 1] * interp
        }
    }
}
