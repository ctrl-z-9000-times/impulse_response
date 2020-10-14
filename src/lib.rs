/*! # Impulse Response Integration Method

A tool for simulating dynamical systems, specializing in large and sparsely
connected systems.

This method works by measuring the impulse response of the system. If the system
is linear and time-invariant then the impulse response completely describes the
system. The exact state of the system can be computed by convolving the initial
state of the system with the impulse response. This method uses these facts to
solve initial value integration problems with efficient sparse matrix-vector
multiplications.

### Literature

Rotter, S., Diesmann, M. Exact digital simulation of time-invariant linear
systems with applications to neuronal modeling. Biol Cybern 81, 381–402 (1999).
<https://doi.org/10.1007/s004220050570>

### Details

Users specify their system with both:

* A state vector, which completely describes the system at an instant in time.

* The derivative of the state, as a function of the current state.

The users system is *assumed* to be linear and time-invariant.

This method does not allow external inputs. Instead the user should directly
modify the states to account for any inputs.

This method uses the impulse response to advance the state of the system in
fixed time steps of length `time_step`. First compute the impulse response in
high fidelity using the Crank-Nicolson method with a variable length time-step.
Sample the response at `time_step` after the impulse. Measure the impulse
response of every state and store them in a matrix. Then to advance the state of
the integration, multiply that matrix by the state vector.

The impulse response matrix is a square matrix, and so its size is the length of
the state vector squared. Naively, this could cause performance issues for
systems which have a very large state vector. However in most systems with very
large state vectors: most of the states do not interact with each other over the
relatively short `time_step` at which it measures the impulse response. As a
result, the impulse responses are mostly zeros and the impulse response matrix
can be compressed into a sparse matrix.

The impulse response integration method runs fast, but can consume a significant
amount of time and memory at start up to compute and store the impulse
responses.

## Example: Measuring equivalent resistance

This comic strip poses an interesting problem. The problem does have a known
analytic solution, `4/pi - 1/2`, but it can also be approximated using numerical
methods. I demonstrate how to do this using the impulse response library.

[![](https://imgs.xkcd.com/comics/nerd_sniping.png)](https://xkcd.com/356/)

### Numerical Solution

* First alter the size of the grid of resistors, from an infinite grid to a very
large grid. Otherwise it would not be possible to compute! Because of this
change the resulting approximation will overestimate the true value. In the
limit, as the size of the grid approaches infinity, this overestimation error
approaches zero.

* Attach a capacitor to every node in the grid. This simulates the parasitic
capacitance which exists in all wires. Capacitance needs to be included in the
model in order to simulate the flow of electric charge.

* Connect a voltage source across the two marked nodes, and measure how much
current passes through the voltage source. Then compute the equivalent
resistance using the formula: `V = I R`. Since the system contains capacitors,
it will take time for the current to reach a steady state. Measuring the steady
state current entails simulating the system for a considerable amount of time.

### Implementation and Results

The source code is an annotated example of how to use this library.
Link: [impulse_response/examples/nerd_sniping.rs](https://github.com/ctrl-z-9000-times/impulse_response/blob/master/examples/nerd_sniping.rs)

Result of running the code with a 32x32 grid:
```text
$ time cargo run --example nerd_sniping --release
Model Size: 1024 Nodes
Equivalent Resistance: 0.8825786612296072 Ohms
Exact Answer: 4/PI - 1/2 = 0.7732395447351628 Ohms
```

Now lets increase the size of the grid to 633x633, and observe that the
equivalent resistance is closer to the correct value:
```text
$ cargo run --example nerd_sniping --release
Model Size: 400689 Nodes
Equivalent Resistance: 0.8416329950362197 Ohms
Exact Answer: 4/PI - 1/2 = 0.7732395447351628 Ohms
```
Runtime: 2 days.

The measurement error is approximately 8%.

## More Examples

* `tests/leaky_cable.rs`
    + Simulates the electricity in a neurons dendrite.

* `tests/conservation.rs`
    + An artificial scenario.
    + Demonstrates modifying a system while its running.
*/

#![feature(trait_alias)]
#![allow(clippy::needless_return)]

use rayon::prelude::*;

/** Main class */
pub struct Model {
    /// `ImpulseResponseMatrix[destination, source]`
    irm: SparseMatrix,
    _touched: Vec<usize>,
    time_step: f64,
    cutoff: f64,
    error_tolerance: f64,
    min_dt: f64,
}

const SUBDIVIDE: usize = 2;

impl Model {
    /**
    + `time_step`: The simulation proceeds in fixed increments of this time.

    + `error_tolerance`: The model will attempt to yield results within this
    error tolerance of the true value of the equations, per unit time. Note:
    errors may accumulate over multiple a long period of time. The error is
    computed as the maximum absolute difference between the approximate and
    true states.

    + `min_dt`: Overrides the variable time step of the numeric integration to
    keep it moving forward at an acceptable speed. This override may compromise
    the `error_tolerance` parameter.

    + `cutoff`: In order to keep the impulse response matrix sparse, all values
    less than the `cutoff` are set to zero. Use `f64::EPSILON`.
    */
    pub fn new(time_step: f64, error_tolerance: f64, min_dt: f64, cutoff: f64) -> Model {
        assert!(time_step > 0.0);
        assert!(error_tolerance > 0.0);
        assert!(min_dt > 0.0);
        assert!(min_dt <= time_step);
        assert!(cutoff >= 0.0);
        Model {
            time_step,
            error_tolerance,
            min_dt,
            cutoff,
            _touched: vec![],
            irm: Default::default(),
        }
    }

    /** Number of points in the model. */
    pub fn len(&self) -> usize {
        let mut len = self.irm.len();
        for point in self.touched() {
            len = len.max(*point + 1);
        }
        return len;
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /** Fraction of values in the Impulse Response Matrix which are Non-Zero. */
    pub fn density(&self) -> f64 {
        return self.irm.data.len() as f64 / self.irm.len().pow(2) as f64;
    }

    /** Add or Update a point.

    Users must call this method when a point is added, or the structure of the
    system at a point is modified.

    Points in the simulation are identified by an index into an array. */
    pub fn touch(&mut self, point: usize) {
        self._touched.push(point)
    }

    pub fn touched(&self) -> &[usize] {
        &self._touched
    }

    /** Run the model forward by one `time_step`. */
    pub fn advance(
        &mut self,
        current_state: &[f64],
        next_state: &mut [f64],
        derivative: impl Derivative,
    ) {
        if !self.touched().is_empty() {
            self.update_irm(derivative);
        }
        self.irm.x_vector(current_state, next_state);
    }

    fn update_irm(&mut self, derivative: impl Derivative) {
        // Recompute all points which were touched, or can interact with a
        // touched point (`IRM[touched, point] != 0`).
        for t in 0..self._touched.len() {
            let touched_point = unsafe { *self._touched.get_unchecked(t) };
            if touched_point < self.irm.len() {
                let row_start = self.irm.row_ranges[touched_point];
                let row_end = self.irm.row_ranges[touched_point + 1];
                self._touched
                    .extend_from_slice(&self.irm.column_indices[row_start..row_end]);
            } else {
                self.irm.resize(touched_point + 1);
            }
        }
        self._touched.par_sort_unstable();
        self._touched.dedup();
        // Measure the impulse response at the touched points.
        let results: Vec<_> = self
            ._touched
            .par_iter()
            .map(|&point| {
                let mut state = SparseVector::new();
                state.insert(point, 1.0);
                state = self.integrate(state, &derivative);
                let mut coordinates: Vec<_> = state.drain().collect();
                // Apply the cutoff. All values below the cutoff are set to
                // zero. The removed values are summed and uniformly
                // redistributed to the remaining non-zero values.
                let mut sum_removed_values = 0.0;
                coordinates.retain(|(_, value)| {
                    if value.abs() < self.cutoff {
                        sum_removed_values += *value;
                        return false;
                    }
                    return true;
                });
                // Redistribute all of the truncated values, so that the sum
                // total of the state is preserved. Uniformly redistribute so
                // as to incur the smallest possible deviation from the
                // original data.
                sum_removed_values /= coordinates.len() as f64;
                for (_, value) in &mut coordinates {
                    *value += sum_removed_values;
                }
                return coordinates;
            })
            .collect();
        // Merge new data into existing IRM.
        self.irm.write_columns(&self._touched, &results);
        self._touched.clear();
    }

    /// Integrate over one time_step.
    #[doc(hidden)]
    pub fn integrate(&self, mut state: SparseVector, derivative: &impl Derivative) -> SparseVector {
        let mut t = 0.0;
        // Run two integrations and compare results. Use different time
        // steps to determine the effect of time step on integration
        // accuracy. If both integrations yield approximately the same
        // output then they're both acceptable, else retry with better
        // time resolution.
        let mut low_res = None;
        let min_dt = self.min_dt * SUBDIVIDE as f64;
        let mut dt = min_dt;
        let mut final_iteration = false;
        while t < self.time_step {
            if dt > self.time_step - t {
                dt = self.time_step - t;
                final_iteration = true;
            }
            if low_res.is_none() {
                low_res = Some(Self::crank_nicolson(state.clone(), derivative, dt));
            }
            let mut first_subdivision_dt = None;
            let mut first_subdivision_state = None;
            let mut high_res = state.clone();
            for i in 0..SUBDIVIDE {
                let high_res_dt = dt / SUBDIVIDE as f64;
                if !high_res_dt.is_normal() {
                    panic!("Failed to find time step which satisfies requested accuracy!")
                }
                high_res = Self::crank_nicolson(high_res, derivative, high_res_dt);
                if i == 0 {
                    first_subdivision_dt = Some(high_res_dt);
                    first_subdivision_state = Some(high_res.clone());
                }
            }
            // Compare low & high time-resolution results.
            let error = max_abs_diff(low_res.as_ref().unwrap(), &high_res);
            let error_ok = error <= self.error_tolerance * dt;
            if dt <= min_dt || error_ok {
                // Accept these results, take the higher accuracy results and continue.
                std::mem::swap(&mut state, &mut high_res);
                low_res.take();
                if final_iteration {
                    break;
                }
                debug_assert!(t + dt > t);
                t += dt;
                if error_ok {
                    dt *= SUBDIVIDE as f64;
                } else if cfg!(debug_assertions) {
                    eprintln!("Warning, max_integration_steps has compromised the accuracy by a factor of {}!",
                        error / self.error_tolerance / dt);
                }
            } else {
                dt = first_subdivision_dt.unwrap();
                if dt >= min_dt {
                    // Reuse the first subdivision of the high_res results
                    // as the next low_res solution.
                    low_res.replace(first_subdivision_state.unwrap());
                } else {
                    dt = min_dt;
                    low_res.take();
                }
            }
        }
        return state;
    }

    /// Do a single discrete step of Crank-Nicolson numeric integration.
    fn crank_nicolson(
        mut state: SparseVector,
        derivative: &impl Derivative,
        dt: f64,
    ) -> SparseVector {
        // Compute the explicit derivative at the current state.
        let mut deriv = SparseVector::with_capacity(state.len() + state.len() / 2);
        derivative(&state, &mut deriv);
        clean_sparse_vector(&mut deriv);
        let iterations = 1;
        for _ in 0..iterations {
            let mut halfway = state.clone();
            add_multiply(&mut halfway, &deriv, dt / 2.0);
            deriv.clear();
            derivative(&halfway, &mut deriv);
            clean_sparse_vector(&mut deriv);
        }
        add_multiply(&mut state, &deriv, dt); // Integrate.
        clean_sparse_vector(&mut state);
        return state;
    }
}

/// State or Derivative Vector, with implicit zeros.
pub type SparseVector = std::collections::HashMap<usize, f64>;

/// Cleans up the nonzero list.
///
/// - Removes zero elements,
fn clean_sparse_vector(x: &mut SparseVector) {
    x.retain(|_, val| *val != 0.0);
    if cfg!(debug_assertions) {
        if !x.values().all(|v| v.is_finite()) {
            panic!("Derivative was not finite!");
        }
    }
}

/// Performs the equation: `A*x + B => B`
/// Where A & B are SparseVectors and x is a scalar.
fn add_multiply(b: &mut SparseVector, a: &SparseVector, x: f64) {
    for (idx, a_value) in a.iter() {
        let b_value = *b.get(idx).unwrap_or(&0.0);
        b.insert(*idx, *a_value * x + b_value);
    }
}

#[doc(hidden)]
pub fn max_abs_diff(a: &SparseVector, b: &SparseVector) -> f64 {
    let mut max: f64 = 0.0;
    for (idx, a_value) in a.iter() {
        max = max.max((a_value - b.get(idx).unwrap_or(&0.0)).abs());
    }
    for (idx, b_value) in b.iter() {
        max = max.max((b_value - a.get(idx).unwrap_or(&0.0)).abs());
    }
    return max;
}

#[derive(Debug)]
struct SparseCoordinate {
    row: usize,
    column: usize,
    value: f64,
}

/// Compressed Sparse Row Matrix.
#[derive(Debug)]
struct SparseMatrix {
    pub data: Vec<f64>,
    pub row_ranges: Vec<usize>,
    pub column_indices: Vec<usize>,
}

impl Default for SparseMatrix {
    fn default() -> SparseMatrix {
        SparseMatrix {
            data: vec![],
            column_indices: vec![],
            row_ranges: vec![0],
        }
    }
}

impl SparseMatrix {
    fn len(&self) -> usize {
        self.row_ranges.len() - 1
    }

    fn resize(&mut self, new_size: usize) {
        assert!(new_size >= self.len()); // SparseMatrix can not shrink, can only expand.
        self.row_ranges.resize(new_size + 1, self.data.len());
    }

    fn write_columns(&mut self, columns: &[usize], rows: &[Vec<(usize, f64)>]) {
        let mut delete_columns = vec![false; self.len()];
        for c in columns {
            delete_columns[*c] = true;
        }
        let mut coords = Vec::with_capacity(rows.iter().map(|sv| sv.len()).sum());
        for (c_idx, row) in columns.iter().zip(rows) {
            for (r_idx, value) in row {
                coords.push(SparseCoordinate {
                    row: *r_idx,
                    column: *c_idx,
                    value: *value,
                });
            }
        }
        coords.par_sort_unstable_by(|a, b| a.row.cmp(&b.row));
        let mut insert_iter = coords.iter().peekable();
        let mut result = SparseMatrix::default();
        let max_new_len = self.data.len() + coords.len();
        result.data.reserve(max_new_len);
        result.column_indices.reserve(max_new_len);
        result.row_ranges.reserve(self.row_ranges.len());
        for (row, (row_start, row_end)) in self
            .row_ranges
            .iter()
            .zip(self.row_ranges.iter().skip(1))
            .enumerate()
        {
            // Filter out the existing data from all of the columns which are
            // being written to.
            for index in *row_start..*row_end {
                let column = self.column_indices[index];
                if !delete_columns[column] {
                    result.data.push(self.data[index]);
                    result.column_indices.push(column);
                }
            }
            // Write the new data for the columns.
            while insert_iter.peek().is_some() && insert_iter.peek().unwrap().row == row {
                let coord = insert_iter.next().unwrap();
                result.data.push(coord.value);
                result.column_indices.push(coord.column);
            }
            result.row_ranges.push(result.data.len());
        }
        std::mem::swap(self, &mut result);
    }

    /// Matrix * Vector Multiplication.
    ///
    /// Computes: `self * src => dst`.
    /// Arguments src & dst are dense column vectors.
    fn x_vector(&self, src: &[f64], dst: &mut [f64]) {
        assert!(src.len() == self.len());
        assert!(dst.len() == self.len());
        dst.par_iter_mut().enumerate().for_each(|(row, dst)| {
            let row_start = self.row_ranges[row];
            let row_end = self.row_ranges[row + 1];
            const V: usize = 4; // V is for auto-vectorization
            let mut sums = [0.0; V];
            let mut chunk = row_start;
            if let Some(row_end_chunk) = row_end.checked_sub(V - 1) {
                while chunk < row_end_chunk {
                    for offset in 0..V {
                        let index = chunk + offset;
                        unsafe {
                            sums[offset] += self.data.get_unchecked(index)
                                * src.get_unchecked(*self.column_indices.get_unchecked(index));
                        }
                    }
                    chunk += V;
                }
            }
            for index in chunk..row_end {
                unsafe {
                    sums[0] += self.data.get_unchecked(index)
                        * src.get_unchecked(*self.column_indices.get_unchecked(index));
                }
            }
            *dst = sums.iter().sum();
        });
    }
}

/** Find the derivative of the state at each point with respect to time.

Users provide this function to define how their system works.

+ The first argument is the current state of the model.

+ The second argument is a mutable reference to the derivative, which should be
updated by this function. The derivative is always zeroed before it is given to
this function */
pub trait Derivative = Fn(&SparseVector, &mut SparseVector) + std::marker::Sync;
