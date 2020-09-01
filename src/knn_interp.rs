/*! Interpolation via weighted average of K-Nearest-Neighbors */

// TODO: Take the error function once at initialization, instead of at every
// call to the interpolate method. This way I can use the *users* error function
// to compute the hill climbing error gradient. I think this will help fine tune
// it. Currently hill climbing uses RMS Error.

// TODO: Implement max_distance for interpolations. Simultaneously enable the
// min_k parameter. This will allow it to discard a few outlier interpolation
// points without failing.

use crate::kd_tree::KDTree;
use rand::prelude::*;
use rayon::prelude::*;
use std::fmt;
use std::sync::RwLock;

/// Python: `1 / sum(exp(-x) for x in range(1, 100))`
const MAGIC_INCREMENT: f64 = 1.7182818284590458;

pub struct KnnInterpolator<const DIMS: usize, const PAYLOAD: usize> {
    inner: RwLock<Inner<DIMS, PAYLOAD>>,
}

struct Inner<const DIMS: usize, const PAYLOAD: usize> {
    sample_alpha: f64,
    minimum_sample_fraction: f64,
    // TODO: make sample_fraction an atomic variable in the outter structure, to
    // avoid write-locking when on the golden path. Only write-lock when adding points.
    sample_fraction: f64,
    params: Parameters<DIMS>,
    kd_tree: KDTree<DIMS>,
    outputs: Vec<[f64; PAYLOAD]>,
}

/// These parameters improved the interpolation.
#[derive(Debug, Copy, Clone, PartialEq)]
struct Parameters<const DIMS: usize> {
    /// Number of neighbors to return to the user.
    k: usize,

    /// Minimum number of neighbors needed for interpolation.
    min_k: usize,

    /// Each input dimension has its own units.
    scale_factor: [f64; DIMS],

    /// Max distance between points before they're not neighbors.
    max_distance: f64,
}

impl<const DIMS: usize, const PAYLOAD: usize> KnnInterpolator<DIMS, PAYLOAD> {
    pub fn new(sample_period: f64, minimum_sample_fraction: f64) -> Self {
        Self {
            inner: RwLock::new(Inner {
                sample_alpha: f64::exp(-1.0 / sample_period),
                minimum_sample_fraction,
                sample_fraction: std::f64::consts::E,
                kd_tree: Default::default(),
                outputs: vec![],
                params: Parameters {
                    k: DIMS + 2,
                    min_k: DIMS + 2,
                    scale_factor: [1.0; DIMS],
                    max_distance: f64::INFINITY,
                },
            }),
        }
    }

    /// Number of interpolation points.
    pub fn len(&self) -> usize {
        return self.inner.read().unwrap().kd_tree.len();
    }

    /// Fraction of calls to interpolate which will be double checked.
    pub fn sample_fraction(&self) -> f64 {
        return self.inner.read().unwrap().sample_fraction.min(1.0);
    }

    /// Useful for testing the quality of the interpolation.
    pub fn disable_double_check(&self) {
        self.inner.write().unwrap().sample_fraction = 0.0;
    }

    pub fn assemble(kd_tree: &mut KDTree<DIMS>, data: &mut Vec<[f64; PAYLOAD]>) -> Self {
        let x = Self {
            inner: RwLock::new(Inner {
                kd_tree: Default::default(),
                outputs: Default::default(),
                params: Parameters {
                    k: DIMS + 2,
                    min_k: DIMS + 2,
                    scale_factor: [1.0; DIMS],
                    max_distance: f64::INFINITY,
                },
                sample_alpha: 0.0,
                sample_fraction: 0.0,
                minimum_sample_fraction: 0.0,
            }),
        };
        let mut inner = x.inner.write().unwrap();
        std::mem::swap(&mut inner.kd_tree, kd_tree);
        std::mem::swap(&mut inner.outputs, data);
        inner.hill_climb();
        std::mem::drop(inner);
        return x;
    }

    pub fn disassemble(self, kd_tree: &mut KDTree<DIMS>, data: &mut Vec<[f64; PAYLOAD]>) {
        let mut inner = self.inner.into_inner().unwrap();
        assert!(inner.sample_fraction == 0.0);
        std::mem::swap(&mut inner.kd_tree, kd_tree);
        std::mem::swap(&mut inner.outputs, data);
    }

    /// Interpolates the payload at any given point.
    ///
    /// Argument `function`: computes the exact payload for this point.
    /// Interpolate may or may not call this function.
    ///
    /// Argument `error(approx, exact)` determines if two payloads are equivalent (return Ok)
    /// or if the differences are significant (return Err). This is used to
    /// double check the results and improve the interpolation if necessary.
    pub fn interpolate(
        &self,
        point: &[f64; DIMS],
        function: impl Fn(&[f64; DIMS]) -> [f64; PAYLOAD],
        error: impl Fn(&[f64; PAYLOAD], &[f64; PAYLOAD]) -> Result<(), ()>,
    ) -> [f64; PAYLOAD] {
        let inner = self.inner.read().unwrap();
        let mut exact = rand::random::<f64>() < inner.sample_fraction;
        let approx_value = match inner.interpolate_inner(point, &inner.params, false) {
            Ok(v) => v,
            Err(()) => {
                exact = true;
                [-f64::INFINITY; PAYLOAD]
            }
        };
        std::mem::drop(inner);
        if !exact {
            return approx_value;
        }
        let exact_value = function(point);
        let mut inner = self.inner.write().unwrap();
        match error(&approx_value, &exact_value) {
            Ok(()) => {
                inner.sample_fraction = inner
                    .minimum_sample_fraction
                    .max(inner.sample_fraction * inner.sample_alpha);
            }
            Err(()) => {
                inner.add_point(point, &exact_value);
                inner.sample_fraction += MAGIC_INCREMENT;
            }
        }
        return exact_value;
    }
}

impl<const DIMS: usize, const PAYLOAD: usize> Inner<DIMS, PAYLOAD> {
    fn add_point(&mut self, point: &[f64; DIMS], datum: &[f64; PAYLOAD]) {
        debug_assert!(point.iter().all(|x| x.is_finite()));
        let index = self.kd_tree.len();
        self.kd_tree.touch(index, point);
        self.kd_tree.update();
        self.outputs.push(*datum);
        if index >= DIMS + 42 && index == index.next_power_of_two() {
            self.hill_climb();
        }
    }

    fn interpolate_inner(
        &self,
        point: &[f64; DIMS],
        params: &Parameters<DIMS>,
        discard_closest: bool,
    ) -> Result<[f64; PAYLOAD], ()> {
        let mut nearest_neighbors = self.kd_tree.nearest_neighbors(
            params.k,
            point,
            &|a, b| params.distance(a, b),
            params.max_distance,
        );
        let nearest_neighbors = if discard_closest {
            let len = nearest_neighbors.len();
            &mut nearest_neighbors[1..len]
        } else {
            &mut nearest_neighbors
        };
        if nearest_neighbors.len() < self.params.min_k {
            return Err(());
        }
        // Transform the distances into interpolation weights.
        for (_, distance) in nearest_neighbors.iter_mut() {
            *distance = 1.0 / distance.max(f64::EPSILON); // Do not divide by zero.
        }
        let sum: f64 = nearest_neighbors.iter().map(|(_, weight)| weight).sum();
        // Interpolate.
        let mut result = [0.0; PAYLOAD];
        for (location, weight) in nearest_neighbors.iter() {
            let payload = self.outputs[*location];
            for (out, value) in result.iter_mut().zip(payload.iter()) {
                *out += value * weight / sum;
            }
        }
        return Ok(result);
    }

    fn interpolation_error(&self, selected_points: &[usize], params: &Parameters<DIMS>) -> f64 {
        let mut errors = Vec::with_capacity(selected_points.len());
        // Increase K, the number of interpolation points, by one because
        // the selected point will be discarded.
        let mut params = params.clone();
        params.k += 1;
        for point_index in selected_points {
            let exact_value = self.outputs[*point_index];
            let approx_value =
                self.interpolate_inner(&self.kd_tree.coordinates[*point_index], &params, true);
            errors.push(match approx_value {
                Ok(approx_value) => {
                    // Root Mean Square Error.
                    f64::sqrt(
                        approx_value
                            .iter()
                            .zip(exact_value.iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f64>()
                            / PAYLOAD as f64,
                    )
                }
                Err(_) => f64::NAN, // Interpolation failed.
            });
        }
        // Find the mean error by excluding any missing interpolation values.
        let (num_ok, sum_ok) = errors
            .iter()
            .filter(|x| !x.is_nan())
            .enumerate()
            .fold((0, 0.0), |a, b| (b.0, a.1 + b.1));
        let mean_error = if num_ok > 0 {
            sum_ok / num_ok as f64
        } else {
            f64::INFINITY
        };
        // Replace NaNs with 125% of the mean_error.
        let missing_data_penalty = 1.25 * mean_error * (errors.len() - num_ok) as f64;
        return (sum_ok + missing_data_penalty) / errors.len() as f64;
    }

    fn hill_climb(&mut self) {
        let len = self.kd_tree.len();
        let sample_points: Vec<_> = (0..len).choose_multiple(&mut thread_rng(), 100_000.min(len));
        let mut current_error = self.interpolation_error(&sample_points, &self.params);
        for _ in 0..10 {
            let directions = self.params.clone().all_mutations();
            let errors: Vec<f64> = directions
                .par_iter()
                .map(|indiv| self.interpolation_error(&sample_points, indiv))
                .collect();
            let (best_index, best_error) =
                errors
                    .iter()
                    .enumerate()
                    .fold((0, f64::INFINITY), |(idx1, err1), (idx2, err2)| {
                        if *err2 < err1 {
                            (idx2, *err2)
                        } else {
                            (idx1, err1)
                        }
                    });
            if best_error < current_error {
                self.params = directions[best_index];
                current_error = best_error;
            } else {
                break;
            }
        }
    }
}

impl<const DIMS: usize> Parameters<DIMS> {
    /// Manhattan Distance.
    fn distance(&self, a: &[f64; DIMS], b: &[f64; DIMS]) -> f64 {
        let mut sum = 0.0;
        for i in 0..DIMS {
            sum += f64::abs((a[i] - b[i]) * self.scale_factor[i]);
        }
        return sum;
    }

    fn all_mutations(&self) -> Vec<Self> {
        let mut all = vec![];
        for delta in [-3, -2, -1, 1, 2, 3].iter() {
            let mut clone = self.clone();
            if clone.k as isize + delta < 1 {
                continue;
            }
            clone.k = (clone.k as isize + delta).max(1) as usize;
            clone.min_k = (clone.min_k as isize + delta).max(1) as usize;
            all.push(clone);
        }
        // for delta in [-3, -2, -1, 1, 2, 3] {
        //     let mut clone = self.clone();
        //     clone.min_k = ((clone.min_k as isize + delta.max(&1)) as usize).min(clone.k);
        //     all.push(clone);
        // }
        for percent_change in &[
            0.1, 1.0, 10.0, 50.0, 90.0, 95.0, 98.0, 99.5, 100.5, 102.0, 105.0, 110.0, 200.0,
            1_000.0, 10_000.0, 100_000.0,
        ] {
            for dim in 0..DIMS {
                let mut clone = self.clone();
                clone.scale_factor[dim] *= percent_change / 100.0;
                all.push(clone);
            }
            // let mut clone = self.clone();
            // clone.max_distance *= percent_change / 100.0;
            // all.push(clone);
        }
        return all;
    }
}

impl<const DIMS: usize, const PAYLOAD: usize> fmt::Display for KnnInterpolator<DIMS, PAYLOAD> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "KNN Interpolator <DIMS={}, PAYLOAD={}>\n", DIMS, PAYLOAD)?;
        write!(f, "{} Points, ", self.len())?;
        let inner = self.inner.read().unwrap();
        write!(
            f,
            "Error Rate: {} / {} samples.\n",
            inner.sample_fraction - inner.minimum_sample_fraction,
            (-1.0 / f64::ln(inner.sample_alpha)).round(),
        )?;
        // TODO: Consider showing the parameters.
        Ok(())
    }
}
