/** K-Nearest-Neighbors */

pub struct Grid<const DIMS: usize> {
    min: [f64; DIMS],
    max: [f64; DIMS],
    /// Inverse of distance between interpolation points, in each dimension.
    scale: [f64; DIMS],
    /// Number of interpolation points in each dimension, regularly
    /// spaced through the range [min, max].  The total number
    /// of interpolation points is the product of this array.
    num: [usize; DIMS],
}

impl<const DIMS: usize> Grid<DIMS> {
    pub fn new(min: [f64; DIMS], max: [f64; DIMS], num: [usize; DIMS]) -> Grid<DIMS> {
        let mut scale = [0.0; DIMS];
        for i in 0..DIMS {
            if max[i] > min[i] {
                scale[i] = (num[i] - 1) as f64 / (max[i] - min[i]);
            } else {
                scale[i] = 0_f64;
            }
        }
        return Grid {
            min,
            max,
            num,
            scale,
        };
    }

    pub fn all_point_coordinates(&self) -> Vec<[f64; DIMS]> {
        let mut points = Vec::with_capacity(2_usize.pow(DIMS as u32));
        let num_points = self.num.iter().product();
        for flat_idx in 0..num_points {
            points.push(self.coords_to_point(self.index_to_coords(flat_idx)));
        }
        return points;
    }

    fn index_to_coords(&self, idx: usize) -> [usize; DIMS] {
        let mut coords = [0; DIMS];
        let mut cursor = idx;
        for dim in (0..DIMS).rev() {
            coords[dim] = cursor % self.num[dim];
            cursor /= self.num[dim];
        }
        return coords;
    }

    fn coords_to_index(&self, coords: [usize; DIMS]) -> usize {
        let mut idx = 0;
        for dim in (0..DIMS).rev() {
            idx *= self.num[dim];
            idx += coords[dim];
        }
        return idx;
    }

    fn coords_to_point(&self, mut coords: [usize; DIMS]) -> [f64; DIMS] {
        let mut point = [0.0; DIMS];
        for dim in 0..DIMS {
            point[dim] = self.min[dim] + coords[dim] as f64 / self.scale[dim];
        }
        return point;
    }

    fn bounds_check(&self, point: [f64; DIMS]) -> bool {
        let lower = point.iter().zip(self.min.iter()).all(|(pt, x)| pt >= x);
        let upper = point.iter().zip(self.max.iter()).all(|(pt, x)| pt <= x);
        lower && upper
    }

    // TODO: return errors instead of garbage data.
    pub fn get(&self, point: [f64; DIMS]) -> Vec<(usize, f64)> {
        if !self.bounds_check(point) {
            return vec![];
        }
        let mut lower = [0_usize; DIMS];
        let mut upper = [0_usize; DIMS];
        let mut lower_weight = [0_f64; DIMS];
        let mut upper_weight = [0_f64; DIMS];
        for i in 0..DIMS {
            let local = (point[i] - self.min[i]) * self.scale[i];
            lower[i] = local as usize;
            upper[i] = lower[i] + 1;
            if upper[i] >= self.num[i] {
                lower[i] -= 1;
                upper[i] -= 1;
            }
            lower_weight[i] = (upper[i] as f64) - local;
            upper_weight[i] = local - (lower[i] as f64);
        }
        let NUM_CORNERS: usize = 2_usize.pow(DIMS as u32);
        let mut points = Vec::with_capacity(NUM_CORNERS);
        for idx in 0..NUM_CORNERS {
            let mut coords = [0_usize; DIMS];
            let mut weight = 1_f64;
            for dim in 0..DIMS {
                if idx & (1 << dim) == 0 {
                    coords[dim] = lower[dim];
                    weight *= lower_weight[dim];
                } else {
                    coords[dim] = upper[dim];
                    weight *= upper_weight[dim];
                }
            }
            points.push((self.coords_to_index(coords), weight));
        }
        return points;
    }
}
