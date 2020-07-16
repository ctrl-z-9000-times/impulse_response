/*! Impulse Response Integration Method

Simulate linear time-invariant systems. This method works by measuring the
impulse response of the system, which is computed using the Crank-Nicholson
method with a variable time-step. */
#![feature(trait_alias)]
#[cfg(feature = "python")]
mod python;
mod sparse;
use sparse::SparseMatrix;
pub use sparse::SparseVector;

/** Main class

In this context, sparse means that there are many states and most of them do
not interact with each other. */
pub struct SparseModel {
    /// `ImpulseResponseMatrix[destination, source]`
    irm: SparseMatrix,
    touched: Vec<usize>,
    delta_time: f64,
    method: IntegrationMethod,
    timestep: IntegrationTimestep,
}

enum IntegrationMethod {
    #[allow(dead_code)]
    ForwardEuler,

    #[allow(dead_code)]
    BackwardEuler {
        iterations: usize,
    },

    CrankNicholson {
        iterations: usize,
    },
}

enum IntegrationTimestep {
    #[allow(dead_code)]
    Constant(f64),

    Variable {
        error_tolerance: f64,
    },
}

impl SparseModel {
    /**
    + `delta_time`: The simulation proceeds in fixed increments of this time.

    + `error_tolerance`: The model will attempt to yield results within this
    error tolerance of the true value of the equations, after each time step of
    length delta_time. Note: errors may accumulate over multiple time steps. The
    error is computed as the maximum absolute difference between the approximate
    and true states */
    pub fn new(delta_time: f64, error_tolerance: f64) -> SparseModel {
        SparseModel {
            irm: Default::default(),
            touched: vec![],
            delta_time,
            method: IntegrationMethod::CrankNicholson { iterations: 1 },
            timestep: IntegrationTimestep::Variable { error_tolerance },
        }
    }

    /** Number of points in the model. */
    pub fn len(&self) -> usize {
        // TODO: What if there are more touched points which have not yet been processed?
        self.irm.len()
    }

    /** Add or Update a point.

    Users must call this method when a point is added, or the structure of the
    system at a point is modified.

    Points in the simulation are identified by an array index. */
    pub fn touch(&mut self, point: usize) {
        self.touched.push(point)
    }

    /** Run the model forward by `delta_time`. */
    pub fn advance(
        &mut self,
        current_state: &[f64],
        next_state: &mut [f64],
        derivative: impl Derivative,
    ) {
        self.update_irm(derivative);
        self.irm.x_vector(current_state, next_state);
    }

    fn update_irm(&mut self, mut derivative: impl Derivative) {
        if self.touched.is_empty() {
            return;
        }
        // Recompute all points which were touched, or can interact with a
        // touched point (`IRM[touched, point] != 0`).
        let mut touching_touched = self.touched.clone();
        for touched_point in &self.touched {
            if *touched_point < self.len() {
                let row_start = self.irm.row_ranges[*touched_point];
                let row_end = self.irm.row_ranges[*touched_point + 1];
                touching_touched.extend_from_slice(&self.irm.column_indices[row_start..row_end]);
            } else {
                self.irm.resize(touched_point + 1);
            }
        }
        touching_touched.sort();
        touching_touched.dedup();
        self.touched.clear();
        // Measure the impulse response at the touched points.
        let mut results = vec![];
        for point in &touching_touched {
            let mut state = SparseVector::new(self.len());
            state.data[*point] = 1.0;
            state.nonzero.push(*point);
            state = self.integrate(state, &mut derivative);
            results.push(state);
        }
        // Merge new data into existing IRM.
        self.irm.write_columns(&touching_touched, &results);
    }

    /// Integrate the given state for delta_time.
    ///
    /// This deals with the IntegrationTimestep.
    #[doc(hidden)]
    pub fn integrate(
        &self,
        mut state: SparseVector,
        derivative: &mut impl Derivative,
    ) -> SparseVector {
        let mut t = 0.0;
        match self.timestep {
            IntegrationTimestep::Constant(dt) => {
                while t < self.delta_time {
                    let dt = dt.min(self.delta_time - t);
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
                let mut dt = self.delta_time / 1000.0; // Initial guess for integration time step.
                let mut low_res = None;
                while t < self.delta_time {
                    dt = dt.min(self.delta_time - t);
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
                        high_res = self.integrate_timestep(high_res, derivative, high_res_dt);
                        if i == 0 {
                            first_subdivision_dt = Some(high_res_dt);
                            first_subdivision_state = Some(high_res.clone());
                        }
                    }
                    let err = SparseVector::max_abs_diff(&low_res.unwrap(), &high_res);
                    if err <= error_tolerance {
                        state = high_res;
                        t += dt;
                        dt *= 2.0;
                        low_res = None;
                    } else {
                        // Reuse the first subdivision of the high_res results
                        // as the next low_res solution.
                        dt = first_subdivision_dt.unwrap();
                        low_res = first_subdivision_state;
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
        mut state: SparseVector,
        derivative: &mut impl Derivative,
        dt: f64,
    ) -> SparseVector {
        // Compute the explicit derivative at the current state.
        let mut deriv = SparseVector::new(state.data.len());
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
                }
            }
            IntegrationMethod::CrankNicholson { iterations } => {
                debug_assert!(*iterations > 0);
                for _ in 0..*iterations {
                    let mut halfway = state.clone();
                    halfway.add_multiply(&deriv, dt / 2.0);
                    deriv.clear();
                    derivative(&halfway, &mut deriv);
                }
            }
        }
        state.add_multiply(&deriv, dt); // Integrate.
        state.clean();
        return state;
    }
}

/** Find the derivative of the state at each point with respect to time.

Users provide this function to define how their system works.

+ The first argument is the current state of the model.

+ The second argument is a mutable reference to the derivative, which should be
updated by this function. The derivative is always zeroed before it is given to
this function */
pub trait Derivative = FnMut(&SparseVector, &mut SparseVector);
