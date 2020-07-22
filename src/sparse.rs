/*! Large sparsely connected systems

In this context, sparse means that there are many states and most of them do not
interact with each other.

Sparse systems are not allowed to have external inputs. Instead the user should
directly modified the state to account for any inputs. */

use crate::{IntegrationMethod, IntegrationTimestep};
use rayon::prelude::*;

/** Main class */
pub struct Model {
    /// `ImpulseResponseMatrix[destination, source]`
    irm: SparseMatrix,
    touched: Vec<usize>,
    time_step: f64,
    method: IntegrationMethod,
    timestep: IntegrationTimestep,
}

impl Model {
    /**
    + `time_step`: The simulation proceeds in fixed increments of this time.

    + `error_tolerance`: The model will attempt to yield results within this
    error tolerance of the true value of the equations, after each `time_step`.
    Note: errors may accumulate over multiple time steps. The error is computed
    as the maximum absolute difference between the approximate and true states
    */
    pub fn new(time_step: f64, error_tolerance: f64) -> Model {
        Model {
            irm: Default::default(),
            touched: vec![],
            time_step,
            method: IntegrationMethod::CrankNicholson { iterations: 1 },
            timestep: IntegrationTimestep::Variable { error_tolerance },
        }
    }

    /** Number of points in the model. */
    pub fn len(&self) -> usize {
        let mut len = self.irm.len();
        for point in &self.touched {
            len = len.max(*point + 1);
        }
        return len;
    }

    /** Add or Update a point.

    Users must call this method when a point is added, or the structure of the
    system at a point is modified.

    Points in the simulation are identified by an array index. */
    pub fn touch(&mut self, point: usize) {
        self.touched.push(point)
    }

    /** Run the model forward by `time_step`. */
    pub fn advance(
        &mut self,
        current_state: &[f64],
        next_state: &mut [f64],
        derivative: impl Derivative,
    ) {
        self.update_irm(derivative);
        // TODO: FIXME: This is a thread local, meaning that this statement does
        // not reach all of the worker threads to clear their pools too.
        ZERO_POOL.with(|pool| pool.borrow_mut().clear());
        self.irm.x_vector(current_state, next_state);
    }

    fn update_irm(&mut self, derivative: impl Derivative) {
        if self.touched.is_empty() {
            return;
        }
        // Recompute all points which were touched, or can interact with a
        // touched point (`IRM[touched, point] != 0`).
        let mut touching_touched = self.touched.clone();
        for touched_point in &self.touched {
            if *touched_point < self.irm.len() {
                let row_start = self.irm.row_ranges[*touched_point];
                let row_end = self.irm.row_ranges[*touched_point + 1];
                touching_touched.extend_from_slice(&self.irm.column_indices[row_start..row_end]);
            } else {
                self.irm.resize(touched_point + 1);
            }
        }
        touching_touched.par_sort_unstable();
        touching_touched.dedup();
        self.touched.clear();
        // Measure the impulse response at the touched points.
        let results: Vec<_> = touching_touched
            .par_iter()
            .map(|point| {
                let mut state = Vector::new(self.len());
                state.data[*point] = 1.0;
                state.nonzero.push(*point);
                state = self.integrate(state, &derivative);
                return state.to_coordinates();
            })
            .collect();
        // Merge new data into existing IRM.
        self.irm.write_columns(&touching_touched, &results);
    }

    /// Integrate the given state for time_step.
    ///
    /// This deals with the IntegrationTimestep.
    #[doc(hidden)]
    pub fn integrate(&self, mut state: Vector, derivative: &impl Derivative) -> Vector {
        let mut t = 0.0;
        match self.timestep {
            IntegrationTimestep::Constant(dt) => {
                while t < self.time_step {
                    let dt = dt.min(self.time_step - t);
                    state = self.integrate_timestep(state, derivative, dt);
                    t += dt;
                }
            }
            IntegrationTimestep::Variable { error_tolerance } => {
                // Run two integrations and compare results. Use different time
                // steps to determine the effect of time step on integration
                // accuracy. If both integrations yield approximately the same
                // output then they're both acceptable, else retry with better
                // time resolution.
                let mut dt = self.time_step / 1000.0; // Initial guess for integration time step.
                let mut low_res = None;
                while t < self.time_step {
                    dt = dt.min(self.time_step - t);
                    if low_res.is_none() {
                        low_res = Some(self.integrate_timestep(state.clone(), derivative, dt));
                    }
                    let subdivide = 7;
                    let mut first_subdivision_dt = None;
                    let mut first_subdivision_state = None;
                    let mut high_res = state.clone();
                    // TODO: Consider applying a jitter to subdivision boundaries.
                    for i in 0..subdivide {
                        let high_res_dt = dt / subdivide as f64;
                        if high_res_dt == 0.0 {
                            panic!("Failed to find time step which satisfies requested accuracy!")
                        }
                        high_res = self.integrate_timestep(high_res, derivative, high_res_dt);
                        if i == 0 {
                            first_subdivision_dt = Some(high_res_dt);
                            first_subdivision_state = Some(high_res.clone());
                        }
                    }
                    let error = Vector::max_abs_diff(low_res.as_ref().unwrap(), &high_res);
                    if error <= error_tolerance * (dt / self.time_step) {
                        std::mem::swap(&mut state, &mut high_res);
                        high_res.return_to_pool();
                        t += dt;
                        dt *= 2.0;
                        low_res.take().unwrap().return_to_pool();
                        first_subdivision_state.unwrap().return_to_pool();
                    } else {
                        // Reuse the first subdivision of the high_res results
                        // as the next low_res solution.
                        dt = first_subdivision_dt.unwrap();
                        low_res
                            .replace(first_subdivision_state.unwrap())
                            .unwrap()
                            .return_to_pool();
                        high_res.return_to_pool();
                    }
                }
            }
        };
        return state;
    }

    /// Do a single discrete step of the numeric integration.
    ///
    /// This deals with the IntegrationMethod.
    fn integrate_timestep(
        &self,
        mut state: Vector,
        derivative: &impl Derivative,
        dt: f64,
    ) -> Vector {
        // Compute the explicit derivative at the current state.
        let mut deriv = Vector::new(state.data.len());
        derivative(&state, &mut deriv);
        // Backward Euler & Crank Nicholson methods calculate the implicit
        // derivative at a future state and override the explicit derivative.
        match &self.method {
            IntegrationMethod::ForwardEuler => {}
            IntegrationMethod::BackwardEuler { iterations } => {
                debug_assert!(*iterations > 0);
                for _ in 0..*iterations {
                    let mut end_state = state.clone();
                    end_state.add_multiply(&deriv, dt);
                    deriv.clear();
                    derivative(&end_state, &mut deriv);
                    end_state.return_to_pool();
                }
            }
            IntegrationMethod::CrankNicholson { iterations } => {
                debug_assert!(*iterations > 0);
                for _ in 0..*iterations {
                    let mut halfway = state.clone();
                    halfway.add_multiply(&deriv, dt / 2.0);
                    deriv.clear();
                    derivative(&halfway, &mut deriv);
                    halfway.return_to_pool();
                }
            }
        }
        state.add_multiply(&deriv, dt); // Integrate.
        deriv.return_to_pool();
        state.clean();
        return state;
    }
}

/** Container for a vector which tracks its non-zero elements. */
#[derive(Debug)]
pub struct Vector {
    /** Dense array of all elements in the vector, including any zeros.

    The user is responsible for updating the `nonzero` list! Check data elements
    before overwriting them, and if you change a zero to a non-zero value then
    append the index to the `nonzero` list. */
    pub data: Vec<f64>,

    /** Indices of the non-zero elements in `data`.

    - May be unsorted,
    - May contain duplicates,
    - May refer to elements with a value of zero.
    */
    pub nonzero: Vec<usize>,
}

thread_local!(static ZERO_POOL: std::cell::RefCell<Vec<Vector>> = std::cell::RefCell::new(vec![]));

impl Vector {
    #[doc(hidden)]
    pub fn new(size: usize) -> Self {
        match ZERO_POOL.with(|pool| pool.borrow_mut().pop()) {
            Some(mut available) => {
                available.clear();
                if available.data.len() != size {
                    available.data.resize(size, 0.0)
                }
                available
            }
            None => Vector {
                data: vec![0.0; size],
                nonzero: vec![],
            },
        }
    }

    fn return_to_pool(self) {
        ZERO_POOL.with(|pool| pool.borrow_mut().push(self))
    }

    fn clear(&mut self) {
        for idx in &self.nonzero {
            self.data[*idx] = 0.0;
        }
        self.nonzero.clear();
    }

    /// Cleans up the nonzero list
    ///
    /// - Sorts and deduplicates,
    /// - Removes zero elements,
    /// - Checks for the inclusion of all non-zero elements.
    fn clean(&mut self) {
        self.nonzero.sort_unstable();
        self.nonzero.dedup();
        let data = &self.data;
        self.nonzero.retain(|idx| data[*idx] != 0.0);
        // Check that nonzero contains *all* non-zero elements.
        debug_assert!(self
            .data
            .iter()
            .enumerate()
            .all(|(i, v)| *v == 0.0 || self.nonzero.binary_search(&i).is_ok()));
    }

    /// Performs the equation: `A*x + B => B`
    /// Where A & B are SparseVectors and x is a scalar.
    fn add_multiply(&mut self, a: &Vector, x: f64) {
        debug_assert_eq!(a.data.len(), self.data.len());
        for point in &a.nonzero {
            let value = &mut self.data[*point];
            if *value == 0.0 {
                self.nonzero.push(*point);
            }
            *value += a.data[*point] * x;
        }
    }

    #[doc(hidden)]
    pub fn max_abs_diff(a: &Vector, b: &Vector) -> f64 {
        debug_assert_eq!(a.data.len(), b.data.len());
        let mut max: f64 = 0.0;
        for idx in &a.nonzero {
            max = max.max((a.data[*idx] - b.data[*idx]).abs());
        }
        for idx in &b.nonzero {
            max = max.max((a.data[*idx] - b.data[*idx]).abs());
        }
        return max;
    }

    fn to_coordinates(self) -> Vec<(usize, f64)> {
        let mut coords = Vec::with_capacity(self.nonzero.len());
        for idx in &self.nonzero {
            coords.push((*idx, self.data[*idx]));
        }
        self.return_to_pool();
        return coords;
    }
}

impl Clone for Vector {
    fn clone(&self) -> Self {
        let mut x = Vector::new(self.data.len());
        x.nonzero = self.nonzero.clone();
        for idx in &self.nonzero {
            x.data[*idx] = self.data[*idx];
        }
        return x;
    }
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
pub trait Derivative = Fn(&Vector, &mut Vector) + std::marker::Sync;
