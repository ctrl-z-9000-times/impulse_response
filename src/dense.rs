/*! TODO: Docs */

use crate::{IntegrationMethod, IntegrationTimestep};
use std::convert::TryInto;
use std::fmt;
use std::sync::{Arc, RwLock};

/// Returns None if Elapsed Ticks == 0, No tables used.
fn max_table_idx(elapsed_ticks: u32) -> Option<usize> {
    if elapsed_ticks <= 0 {
        None
    } else {
        Some((32 - 1 - elapsed_ticks.leading_zeros()) as usize)
    }
}

const INTERP_SAMPLE_PERIOD: f64 = 1e4;
const INTERP_MIN_SAMPLE_FRACTION: f64 = 1e-4;
const SCHED_SAMPLE_PERIOD: f64 = 1e4;
const SCHED_MIN_SAMPLE_FRACTION: f64 = 1e-4;
/// Python: `1 / sum(exp(-x) for x in range(1, 100))`
const MAGIC_INCREMENT: f64 = 1.7182818284590458;

/** TODO: Docs */
pub struct Model<
    const INPUTS: usize,
    const OUTPUTS: usize,
    const STATES: usize,
    const IRM: usize,
    const INPUTS_STATES: usize,
> {
    pub time_step: f64,
    pub derivative_function: Box<dyn Fn(&[f64; INPUTS], &[f64; STATES]) -> [f64; STATES]>,
    pub output_function: Box<dyn Fn(&[f64; STATES]) -> [f64; OUTPUTS]>,
    pub invariant_function: Option<Box<dyn Fn(&mut [f64; STATES])>>,
    pub max_input_error: [f64; INPUTS],
    pub max_output_error: [f64; OUTPUTS],
    pub max_state_error: [f64; STATES],
    pub advance_data: RwLock<Vec<crate::knn::KDTree<INPUTS, IRM>>>,
    pub interp_sample_fraction: RwLock<f64>,
    pub sched_sample_fraction: RwLock<f64>,
    pub scheduler: RwLock<Scheduler<INPUTS_STATES>>,
}

pub struct Scheduler<const INPUTS_STATES: usize> {
    pub scheduler_data: crate::knn::KDTree<INPUTS_STATES, 1>,
    pub time_step_too_long: bool,
    pub undersleep_factor: hdrhistogram::Histogram<u64>,
    pub oversleep_error: hdrhistogram::Histogram<u64>,
}

/** TODO: Docs */
#[derive(Clone)]
pub struct Instance<
    const INPUTS: usize,
    const OUTPUTS: usize,
    const STATES: usize,
    const IRM: usize,
    const INPUTS_STATES: usize,
> {
    pub model: Arc<Model<INPUTS, OUTPUTS, STATES, IRM, INPUTS_STATES>>,
    pub state: [f64; STATES],
    pub previous_inputs: [f64; INPUTS],
    pub previous_outputs: [f64; OUTPUTS],
    pub input_error: [f64; INPUTS],
    pub last_compute: u32,
    pub due_date: u32,
}

#[derive(Copy, Clone)]
struct AdvanceArguments<const INPUTS: usize, const STATES: usize> {
    state: [f64; STATES],
    inputs: [f64; INPUTS],
    elapsed_ticks: u32,
}

impl<
        const INPUTS: usize,
        const OUTPUTS: usize,
        const STATES: usize,
        const IRM: usize,
        const INPUTS_STATES: usize,
    > Instance<INPUTS, OUTPUTS, STATES, IRM, INPUTS_STATES>
{
    /** TODO: Docs */
    pub fn new(
        model: &Arc<Model<INPUTS, OUTPUTS, STATES, IRM, INPUTS_STATES>>,
        initial_state: &[f64],
    ) -> Instance<INPUTS, OUTPUTS, STATES, IRM, INPUTS_STATES> {
        debug_assert!(initial_state.iter().all(|x| x.is_finite()));
        Instance {
            model: model.clone(),
            state: initial_state.try_into().unwrap(),
            previous_inputs: [0.0; INPUTS],
            previous_outputs: [0.0; OUTPUTS],
            input_error: [0.0; INPUTS],
            last_compute: 0,
            due_date: 1,
        }
    }

    pub fn advance_numeric(&mut self, inputs: [f64; INPUTS]) -> [f64; OUTPUTS] {
        debug_assert!(inputs.iter().all(|x| x.is_finite()));
        self.model
            .integrate_timestep(&inputs, &mut self.state, self.model.time_step);
        return (self.model.output_function)(&self.state);
    }

    /** TODO: Docs */
    pub fn advance(&mut self, inputs: [f64; INPUTS]) -> [f64; OUTPUTS] {
        debug_assert!(inputs.iter().all(|x| x.is_finite()));
        self.due_date -= 1;
        self.last_compute += 1;
        let mut input_event = false;
        for i in 0..INPUTS {
            self.input_error[i] += (self.previous_inputs[i] - inputs[i]) * self.model.time_step;
            if self.input_error[i].abs() <= self.model.max_input_error[i] {
                input_event = true;
                break;
            }
        }
        if !input_event && self.due_date > 0 {
            return self.previous_outputs;
        }
        if input_event {
            let mut args_catchup = AdvanceArguments {
                state: self.state,
                inputs: self.previous_inputs,
                elapsed_ticks: self.last_compute - 1,
            };
            if rand::random::<f64>() < *self.model.interp_sample_fraction.read().unwrap() {
                self.model.advance_exact_interpolation(&mut args_catchup);
            } else {
                self.model.advance(&mut args_catchup);
            }
            let mut args_event = AdvanceArguments {
                state: args_catchup.state,
                inputs: inputs,
                elapsed_ticks: 1,
            };
            if rand::random::<f64>() < *self.model.interp_sample_fraction.read().unwrap() {
                self.model.advance_exact_interpolation(&mut args_event);
            } else {
                self.model.advance(&mut args_event);
            }
            self.state = args_event.state;
        } else {
            // Scheduled update.
            let mut args_update = AdvanceArguments {
                state: self.state,
                inputs: self.previous_inputs,
                elapsed_ticks: self.last_compute,
            };
            if rand::random::<f64>() < *self.model.interp_sample_fraction.read().unwrap() {
                self.model.advance_exact_interpolation(&mut args_update);
            } else {
                self.model.advance(&mut args_update);
            }
            self.state = args_update.state
        }
        self.last_compute = 0;
        self.previous_inputs = inputs;
        self.previous_outputs = (self.model.output_function)(&self.state);
        let schedule_exact =
            rand::random::<f64>() < *self.model.sched_sample_fraction.read().unwrap();
        self.due_date = self.model.schedule(&inputs, &self.state, schedule_exact);
        return self.previous_outputs;
    }
}

impl<
        const INPUTS: usize,
        const OUTPUTS: usize,
        const STATES: usize,
        const IRM: usize,
        const INPUTS_STATES: usize,
    > Model<INPUTS, OUTPUTS, STATES, IRM, INPUTS_STATES>
{
    /** TODO: Docs

    Argument `time_step`:

    Argument `derivative_function`:

    Argument `output_function`:

    Argument `invariant_function`:

    Argument `max_input_error`:
        Integral of absolute tolerance.
        dimensions: inputs * time

    Argument `max_output_error`:
        Integral of absolute tolerance.
        dimensions: outputs * time

    Argument `max_state_error`:
        Absolute tolerance per unit of time.
        States are allowed to drift at this rate.
    */
    pub fn new(
        time_step: f64,
        mut derivative_function: Box<dyn Fn(&[f64; INPUTS], &[f64; STATES]) -> [f64; STATES]>,
        mut output_function: Box<dyn Fn(&[f64; STATES]) -> [f64; OUTPUTS]>,
        mut invariant_function: Option<Box<dyn Fn(&mut [f64; STATES])>>,
        max_input_error: &[f64],
        max_output_error: &[f64],
        max_state_error: &[f64],
    ) -> Arc<Model<INPUTS, OUTPUTS, STATES, IRM, INPUTS_STATES>> {
        debug_assert!(STATES * STATES == IRM);
        debug_assert!(INPUTS + STATES == INPUTS_STATES);
        assert!(max_input_error.iter().all(|x| x.is_finite() && *x > 0.0));
        assert!(max_output_error.iter().all(|x| x.is_finite() && *x > 0.0));
        assert!(max_state_error.iter().all(|x| x.is_finite() && *x > 0.0));
        // In debug mode wrap the user-provided functions with error checking.
        if cfg!(debug_assertions) {
            derivative_function = Box::new(move |inputs, state| {
                let derivative = (derivative_function)(inputs, state);
                assert!(derivative.iter().all(|x| x.is_finite()));
                return derivative;
            });
            output_function = Box::new(move |state| {
                let output = (output_function)(state);
                assert!(output.iter().all(|x| x.is_finite()));
                return output;
            });
            if let Some(f) = invariant_function {
                invariant_function = Some(Box::new(move |state| {
                    (f)(state);
                    assert!(state.iter().all(|x| x.is_finite()));
                }));
            }
        }
        return Arc::new(Model {
            time_step,
            derivative_function,
            output_function,
            invariant_function,
            advance_data: RwLock::new(vec![]),
            max_input_error: max_input_error.try_into().unwrap(),
            max_output_error: max_output_error.try_into().unwrap(),
            // TODO: Divide the max_state_error in half BC its used by two
            // mechanisms which both incur this much error?
            max_state_error: max_state_error.try_into().unwrap(),
            interp_sample_fraction: RwLock::new(10.0),
            sched_sample_fraction: RwLock::new(10.0),
            scheduler: RwLock::new(Scheduler {
                scheduler_data: Default::default(),
                time_step_too_long: false,
                undersleep_factor: hdrhistogram::Histogram::new(2).unwrap(),
                oversleep_error: hdrhistogram::Histogram::new(2).unwrap(),
            }),
        });
    }

    /// Integrate the given state for time_step.
    ///
    /// This deals with the IntegrationTimestep.
    fn integrate(&self, inputs: &[f64; INPUTS], state: &mut [f64; STATES]) {
        let mut t = 0.0;
        let timestep_method = IntegrationTimestep::Variable {
            error_tolerance: self.max_state_error[0] / 10.0,
        };
        // let timestep_method = IntegrationTimestep::Constant(1e-6);
        match timestep_method {
            IntegrationTimestep::Constant(dt) => {
                while t < self.time_step {
                    let dt = dt.min(self.time_step - t);
                    self.integrate_timestep(inputs, state, dt);
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
                    debug_assert!(dt > self.time_step * 1e-12);
                    dt = dt.min(self.time_step - t);
                    if low_res.is_none() {
                        low_res = Some(state.clone());
                        self.integrate_timestep(inputs, low_res.as_mut().unwrap(), dt);
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
                        self.integrate_timestep(inputs, &mut high_res, high_res_dt);
                        if i == 0 {
                            first_subdivision_dt = Some(high_res_dt);
                            first_subdivision_state = Some(high_res.clone());
                        }
                    }
                    let error = low_res
                        .as_ref()
                        .unwrap()
                        .iter()
                        .zip(high_res.iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(-f64::INFINITY, f64::max);
                    let max_error = error_tolerance * (dt / self.time_step);
                    if error <= max_error {
                        *state = high_res;
                        t += dt;
                        if error <= max_error / 4.0 {
                            dt *= 1.5;
                        }
                        low_res = None;
                    } else {
                        // Reuse the first subdivision of the high_res results
                        // as the next low_res solution.
                        dt = first_subdivision_dt.unwrap();
                        low_res = first_subdivision_state;
                    }
                }
            }
        }
    }

    /// Do a single discrete step of the numeric integration.
    ///
    /// This deals with the IntegrationMethod.
    fn integrate_timestep(&self, inputs: &[f64; INPUTS], state: &mut [f64; STATES], dt: f64) {
        // Compute the explicit derivative at the current state.
        let mut deriv = (self.derivative_function)(inputs, &state);
        // Backward Euler & Crank Nicholson methods calculate the implicit
        // derivative at a future state and override the explicit derivative.
        let integration_method = IntegrationMethod::CrankNicholson { iterations: 1 };
        // let integration_method = IntegrationMethod::BackwardEuler { iterations: 1 };
        match &integration_method {
            IntegrationMethod::ForwardEuler => {}
            IntegrationMethod::BackwardEuler { iterations } => {
                debug_assert!(*iterations > 0);
                for _ in 0..*iterations {
                    let mut end_state = state.clone();
                    Self::ax_plus_b(&deriv, dt, &mut end_state);
                    deriv = (self.derivative_function)(inputs, &end_state);
                }
            }
            IntegrationMethod::CrankNicholson { iterations } => {
                debug_assert!(*iterations > 0);
                for _ in 0..*iterations {
                    let mut halfway = state.clone();
                    Self::ax_plus_b(&deriv, dt / 2.0, &mut halfway);
                    deriv = (self.derivative_function)(inputs, &halfway);
                }
            }
        }
        Self::ax_plus_b(&deriv, dt, state); // Integrate.
        if let Some(f) = &self.invariant_function {
            (f)(state)
        }
    }

    fn ax_plus_b(a: &[f64; STATES], x: f64, b: &mut [f64; STATES]) {
        for (a_element, b_element) in a.iter().zip(b.iter_mut()) {
            *b_element += a_element * x;
        }
    }

    /// Impulse response matrix for one `time_step`
    fn new_irm(&self, inputs: &[f64; INPUTS]) -> [f64; IRM] {
        let mut matrix = [0.; IRM];
        for src in 0..STATES {
            let mut probe = [0_f64; STATES];
            probe[src] = 1_f64;
            self.integrate(inputs, &mut probe);
            matrix[src * STATES..(src * STATES) + STATES].copy_from_slice(&probe);
        }
        matrix
    }

    /// Doubles the length of time over which this IRM integrates.
    fn square_irm(&self, m: &mut [f64; IRM]) {
        let mut sqr = [0.; IRM];
        for r in 0..STATES {
            for c in 0..STATES {
                for dot in 0..STATES {
                    sqr[r * STATES + c] += m[r * STATES + dot] * m[dot * STATES + c];
                }
            }
        }
        m.copy_from_slice(&sqr);
    }

    fn apply_irm(&self, s: &[f64; STATES], m: &[f64; IRM]) -> [f64; STATES] {
        let mut result = [0f64; STATES];
        for src in 0..STATES {
            for dst in 0..STATES {
                result[dst] += s[src] * m[src * STATES + dst];
            }
        }
        result
    }

    fn advance(&self, args: &mut AdvanceArguments<INPUTS, STATES>) {
        for table_idx in 0..32 {
            if args.elapsed_ticks & (1 << table_idx) == 0 {
                continue;
            } else if args.elapsed_ticks < (1 << table_idx) {
                break;
            }
            if table_idx >= self.advance_data.read().unwrap().len() {
                // Initialize KNN & advance data for this time-scale.
                self.advance_data
                    .write()
                    .unwrap()
                    .resize_with(table_idx + 1, Default::default);
            }
            // Get the nearest input examples for interpolation between IRMs.
            let advance_data_borrow = self.advance_data.read().unwrap();
            let irm = advance_data_borrow[table_idx].interpolate(&args.inputs);
            if let Ok(irm) = irm {
                // Apply the IRM and interpolate between their results.
                args.state = self.apply_irm(&args.state, &irm);
                if let Some(f) = &self.invariant_function {
                    (f)(&mut args.state)
                }
            } else {
                // Nearest neighbors failed. Compute the exact IRM for this point and use it.
                let mut exact_irm = self.new_irm(&args.inputs);
                for _ in 0..table_idx {
                    self.square_irm(&mut exact_irm);
                }
                args.state = self.apply_irm(&args.state, &exact_irm);
                // Add the irm to the nearest neighbors data.
                std::mem::drop(advance_data_borrow);
                self.advance_data.write().unwrap()[table_idx].add_point(&args.inputs, &exact_irm);
            }
        }
    }

    /// Advances and also ensures that the interpolation error is acceptable.
    fn advance_exact_interpolation(
        &self,
        AdvanceArguments {
            state,
            inputs,
            elapsed_ticks,
        }: &mut AdvanceArguments<INPUTS, STATES>,
    ) {
        let max_table_idx = match max_table_idx(*elapsed_ticks) {
            None => return,
            Some(x) => x,
        };
        // Compute the exact results for this constant input & initial state.
        let mut exact_irm = self.new_irm(inputs);
        let mut next_state = state.clone();
        for table_idx in 0..=max_table_idx {
            let ticks = 2_u32.pow(table_idx as u32);
            let exact_state = self.apply_irm(state, &exact_irm);
            if *elapsed_ticks & (1 << table_idx) != 0 {
                next_state = self.apply_irm(&next_state, &exact_irm);
            }
            // Use interpolation to compute the approximate results.
            let mut interpolated_approximation = AdvanceArguments {
                state: *state,
                inputs: *inputs,
                elapsed_ticks: ticks,
            };
            self.advance(&mut interpolated_approximation);
            // Measure the error caused by interpolation.
            let mut sample_fraction = self.interp_sample_fraction.write().unwrap();
            *sample_fraction = INTERP_MIN_SAMPLE_FRACTION
                .max(*sample_fraction * f64::exp(-1.0 / INTERP_SAMPLE_PERIOD));
            if exact_state
                .iter()
                .zip(interpolated_approximation.state.iter())
                .map(|(exact, approx)| (exact - approx).abs())
                .zip(self.max_state_error.iter())
                .any(|(err, max)| err > *max * ticks as f64 * self.time_step)
            {
                // Make this sample into an interpolation point.
                self.advance_data.write().unwrap()[table_idx].add_point(inputs, &exact_irm);
                *sample_fraction += MAGIC_INCREMENT;
            }
            //
            self.square_irm(&mut exact_irm);
        }
        *state = next_state;
    }

    /// Runs a high accuracy simulation of an AdvanceArguments.
    ///
    /// Return pair of (sleep, error)
    ///
    /// + Where sleep is the maximum number of ticks that the sample could
    /// have slept for before exceeding the output error tolerances.
    ///
    /// + Where error is the absolute value of the integral of the error: the
    /// difference between the assumed constant output and the true output over
    /// the specified length of time. The error is normalized into units of
    /// max_output_error, and so the returned error is a multiple of
    /// max_output_error. Each output has an error and the largest error is
    /// returned. An error of less than 1 is an under sleep, and an error of
    /// greater than 1 is an oversleep.
    fn high_accuracy_advance(&self, sample: AdvanceArguments<INPUTS, STATES>) -> (u32, f64) {
        let mut sample_error = None; // Return value.
        let mut correct_sleep = None; // Return value.
        let max_sleep = 2_u32.pow(21);
        let mut cursor = sample;
        let sample_elapsed_ticks = cursor.elapsed_ticks;
        debug_assert!(sample_elapsed_ticks <= max_sleep);
        debug_assert!(sample_elapsed_ticks.is_power_of_two());
        let const_output = (self.output_function)(&cursor.state);
        let mut sleep = 0;
        let mut error_accumulator = [0_f64; OUTPUTS];
        let mut prev_output = const_output;
        while sample_error == None || correct_sleep == None {
            // Advance in exponentially increasing time steps.
            cursor.elapsed_ticks = sleep / 16;
            // Round time step down to the nearest power of 2.
            cursor.elapsed_ticks =
                2_u32.pow(max_table_idx(cursor.elapsed_ticks).unwrap_or(0) as u32);
            //
            self.advance(&mut cursor);
            let true_output = (self.output_function)(&cursor.state);
            for o in 0..OUTPUTS {
                let trapazoid_rule = (true_output[o] + prev_output[o]) / 2.;
                error_accumulator[o] += (const_output[o] - trapazoid_rule)
                    * cursor.elapsed_ticks as f64
                    * self.time_step;
            }
            prev_output = true_output;
            // Check if the output error exceeds its tolerance.
            if correct_sleep == None
                && error_accumulator
                    .iter()
                    .zip(self.max_output_error.iter())
                    .any(|(err, &max)| err.abs() > max)
            {
                correct_sleep = Some(sleep)
            }
            sleep += cursor.elapsed_ticks;
            if correct_sleep == None && sleep >= max_sleep {
                correct_sleep = Some(sleep)
            }
            // Measure the error at the given samples elapsed_ticks.
            if sleep == sample_elapsed_ticks {
                sample_error = Some(
                    error_accumulator
                        .iter()
                        .zip(self.max_output_error.iter())
                        .map(|(err, max)| err.abs() / max)
                        .fold(-f64::INFINITY, |m, err| m.max(err)),
                )
            }
        }
        (correct_sleep.unwrap(), sample_error.unwrap())
    }

    /// Returns the number of ticks until the next compute.
    ///
    /// Argument exact will run a higher accuracy simulation to determine the
    /// correct sleep time.
    fn schedule(&self, inputs: &[f64; INPUTS], state: &[f64; STATES], mut exact: bool) -> u32 {
        // Run the schedulers internal classifier.
        let mut clsr_input = Vec::with_capacity(STATES + INPUTS);
        clsr_input.extend_from_slice(inputs);
        clsr_input.extend_from_slice(state);
        let borrow = self.scheduler.read().unwrap();
        let scheduled_sleep_power = borrow
            .scheduler_data
            .interpolate(clsr_input.as_slice().try_into().unwrap());
        let scheduled_sleep_power = match scheduled_sleep_power {
            Ok(x) => x[0],
            Err(_) => {
                exact = true;
                0.0
            }
        };
        let scheduled_sleep = 2u32.pow(scheduled_sleep_power.floor() as u32);
        if !exact {
            return scheduled_sleep;
        }
        std::mem::drop(borrow);
        // Determine the ground truth by simulating with a shorter time step.
        let (mut correct_sleep_exact, error) = self.high_accuracy_advance(AdvanceArguments {
            inputs: *inputs,
            state: *state,
            elapsed_ticks: scheduled_sleep,
        });
        if correct_sleep_exact < 1 {
            self.scheduler.write().unwrap().time_step_too_long = true;
            correct_sleep_exact = 1;
        }
        // Round down to the nearest power of 2.
        let correct_sleep = 2u32.pow(32 - 1 - correct_sleep_exact.leading_zeros());
        // Update the schedulers internals.
        let mut scheduler = self.scheduler.write().unwrap();
        scheduler
            .oversleep_error
            .record(error.ceil() as u64)
            .unwrap();
        scheduler
            .undersleep_factor
            .record((correct_sleep / scheduled_sleep) as u64)
            .unwrap();
        let mut sample_fraction = self.sched_sample_fraction.write().unwrap();
        *sample_fraction =
            SCHED_MIN_SAMPLE_FRACTION.max(*sample_fraction * f64::exp(-1.0 / SCHED_SAMPLE_PERIOD));
        let scheduled_sleep_exact = 2.0_f64.powf(scheduled_sleep_power);
        let correct_sleep_exact = correct_sleep_exact as f64;
        if scheduled_sleep_exact > correct_sleep_exact
            || scheduled_sleep_exact <= correct_sleep_exact / 2.0
            || !exact
        {
            scheduler.scheduler_data.add_point(
                clsr_input.as_slice().try_into().unwrap(),
                &[correct_sleep_exact.log2()],
            );
        }
        if scheduled_sleep_exact >= correct_sleep_exact * 2.0
            || scheduled_sleep_exact <= correct_sleep_exact / 8.0
            || !exact
        {
            *sample_fraction += MAGIC_INCREMENT;
        }
        return correct_sleep; // Return the more accurately computed results.
    }
}

impl<
        const INPUTS: usize,
        const OUTPUTS: usize,
        const STATES: usize,
        const IRM: usize,
        const INPUTS_STATES: usize,
    > fmt::Display for Model<INPUTS, OUTPUTS, STATES, IRM, INPUTS_STATES>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Dense Impulse Response Model <INPUTS: {}, OUTPUTS: {}, STATES: {}>\n",
            INPUTS, OUTPUTS, STATES
        )?;
        write!(f, "  time_step: {}\n", self.time_step)?;
        if self.scheduler.read().unwrap().time_step_too_long {
            write!(f, "  The time step is too long!\n",)?;
        }
        write!(f, "  max_input_error: {:?}\n", self.max_input_error)?;
        write!(f, "  max_output_error: {:?}\n", self.max_output_error)?;
        write!(f, "  max_state_error: {:?}\n", self.max_state_error)?;
        write!(
            f,
            "Scheduler Error Rate: {} / {} samples.\n",
            *self.sched_sample_fraction.read().unwrap() - SCHED_MIN_SAMPLE_FRACTION,
            SCHED_SAMPLE_PERIOD,
        )?;
        write!(
            f,
            "Scheduler Interpolation Points: {}\n",
            self.scheduler.read().unwrap().scheduler_data.len()
        )?;
        write!(
            f,
            "Scheduler Overslept: {:.6} % of samples ({:.6} % critical).\n",
            100.0
                - self
                    .scheduler
                    .read()
                    .unwrap()
                    .oversleep_error
                    .percentile_below(1),
            100.0
                - self
                    .scheduler
                    .read()
                    .unwrap()
                    .oversleep_error
                    .percentile_below(10)
        )?;
        write!(
            f,
            "Scheduler Underslept: {:.6} % of samples ({:.6} % critical).\n",
            100.0
                - self
                    .scheduler
                    .read()
                    .unwrap()
                    .undersleep_factor
                    .percentile_below(1),
            100.0
                - self
                    .scheduler
                    .read()
                    .unwrap()
                    .undersleep_factor
                    .percentile_below(10)
        )?;
        write!(
            f,
            "Interpolation Error Rate: {} / {} samples.\n",
            *self.interp_sample_fraction.read().unwrap() - INTERP_MIN_SAMPLE_FRACTION,
            INTERP_SAMPLE_PERIOD,
        )?;
        for table_idx in 0..self.advance_data.read().unwrap().len() {
            write!(
                f,
                "  Advance 2^{:>2}: {:>3} input interpolation points.\n",
                table_idx,
                self.advance_data.read().unwrap()[table_idx].len()
            )?;
        }
        return Ok(());
    }
}
