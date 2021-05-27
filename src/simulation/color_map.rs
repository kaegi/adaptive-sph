use crate::{floating_type_mod::FT, V};

pub type Color = V<FT, 3>;

pub struct ColorMap {
    insertions: Vec<(FT, Color)>,
}

impl ColorMap {
    pub fn new(mut insertions: Vec<(FT, Color)>) -> Self {
        insertions.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        Self { insertions }
    }

    pub fn get(&self, x: FT) -> Color {
        if x <= self.insertions[0].0 {
            return self.insertions[0].1;
        }
        if x >= self.insertions.last().unwrap().0 {
            return self.insertions.last().unwrap().1;
        }

        for i in 0..self.insertions.len() - 1 {
            if x >= self.insertions[i].0 && x <= self.insertions[i + 1].0 {
                let interp = (x - self.insertions[i].0) / (self.insertions[i + 1].0 - self.insertions[i].0);
                return self.insertions[i].1 + interp * (self.insertions[i + 1].1 - self.insertions[i].1);
            }
        }

        unreachable!("retrieving color for value {} failed", x)
    }

    pub fn get_u8(&self, x: FT) -> V<u8, 3> {
        self.get(x).map(|f| (f * 255.) as u8)
    }

    pub fn get_f64(&self, x: FT) -> V<f64, 3> {
        self.get(x).map(|f| f as f64)
    }

    pub fn get_f32(&self, x: FT) -> V<f32, 3> {
        self.get(x).map(|f| f as f32)
    }

    pub fn color_stops(&self) -> &[(FT, Color)] {
        &self.insertions
    }
}
