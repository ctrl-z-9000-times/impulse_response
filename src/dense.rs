use crate::{IntegrationMethod, IntegrationTimestep};
use std::convert::{TryFrom, TryInto};
use std::sync::{Arc, Mutex, RwLock};

/// Returns None if Elapsed Ticks == 0, No tables used.
fn max_table_idx(elapsed_ticks: u32) -> Option<usize> {
    if elapsed_ticks <= 0 {
        None
    } else {
        Some((32 - 1 - elapsed_ticks.leading_zeros()).try_into().unwrap())
    }
}

pub struct Model<const INPUTS: usize, const OUTPUTS: usize, const STATES: usize, const IRM: usize> {
    delta_time: f64,
    derivative: Box<dyn Fn(&[f64; INPUTS], &[f64; STATES]) -> [f64; STATES]>,
    output_function: Box<dyn Fn(&[f64; STATES]) -> [f64; OUTPUTS]>,
    max_input_error: [f64; INPUTS],
    max_output_error: [f64; OUTPUTS],
    interpolation_data: Vec<crate::knn::Grid<INPUTS>>,
    advance_data: Vec<Vec<[f64; IRM]>>,
    samples: Mutex<Vec<AdvanceArguments<INPUTS, STATES>>>,
    // TODO: Adjust how many samples this takes based on how well the
    // interpolation & scheduling are working.
    sample_fraction: f64,
    scheduler_data: crate::nn::NN,
    tick_too_long: bool,
    diagnostic_period: f64,
    scheduler_enabled: bool,
    mean_sleep: f64,
    mean_undersleep_factor: f64,
    oversleep_error_histogram: crate::histogram::OversleepErrorHistogram,
}

pub struct Instance<
    const INPUTS: usize,
    const OUTPUTS: usize,
    const STATES: usize,
    const IRM: usize,
> {
    model: Arc<RwLock<Model<INPUTS, OUTPUTS, STATES, IRM>>>,
    state: [f64; STATES],
    previous_inputs: [f64; INPUTS],
    previous_outputs: [f64; OUTPUTS],
    input_error: [f64; INPUTS],
    last_compute: u32,
    due_date: u32,
}

#[derive(Copy, Clone)]
struct AdvanceArguments<const INPUTS: usize, const STATES: usize> {
    state: [f64; STATES],
    inputs: [f64; INPUTS],
    elapsed_ticks: u32,
}

impl<const INPUTS: usize, const OUTPUTS: usize, const STATES: usize, const IRM: usize>
    Instance<INPUTS, OUTPUTS, STATES, IRM>
{
    pub fn new(
        model: Arc<RwLock<Model<INPUTS, OUTPUTS, STATES, IRM>>>,
        initial_state: [f64; STATES],
    ) -> Instance<INPUTS, OUTPUTS, STATES, IRM> {
        Instance {
            model,
            state: initial_state,
            previous_inputs: [0.0; INPUTS],
            previous_outputs: [0.0; OUTPUTS],
            input_error: [0.0; INPUTS],
            last_compute: 0,
            due_date: 1,
        }
    }

    pub fn advance(&mut self, inputs: [f64; INPUTS]) -> [f64; OUTPUTS] {
        self.due_date -= 1;
        self.last_compute += 1;
        let mut input_event = false;
        let model = self.model.read().unwrap();
        for i in 0..INPUTS {
            self.input_error[i] += (self.previous_inputs[i] - inputs[i]) * model.delta_time;
            if self.input_error[i].abs() <= model.max_input_error[i] {
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
            model.advance(&mut args_catchup);
            if rand::random::<f64>() < model.sample_fraction {
                model.samples.lock().unwrap().push(args_catchup)
            }
            let mut args_event = AdvanceArguments {
                state: args_catchup.state,
                inputs: inputs,
                elapsed_ticks: 1,
            };
            model.advance(&mut args_event);
            if rand::random::<f64>() < model.sample_fraction {
                model.samples.lock().unwrap().push(args_event)
            }
            self.state = args_event.state;
        } else {
            // Scheduled update.
            let mut args_update = AdvanceArguments {
                state: self.state,
                inputs: self.previous_inputs,
                elapsed_ticks: self.last_compute,
            };
            model.advance(&mut args_update);
            if rand::random::<f64>() < model.sample_fraction {
                model.samples.lock().unwrap().push(args_update)
            }
            self.state = args_update.state
        }
        self.due_date = if model.scheduler_enabled {
            let mut scheduler_input = Vec::with_capacity(STATES + INPUTS);
            scheduler_input.extend_from_slice(&self.state);
            scheduler_input.extend_from_slice(&inputs);
            let scheduler_output = model.scheduler_data.run(&scheduler_input).pop().unwrap();
            // Self::sleep_time(&scheduler_output)
            1
        } else {
            1
        };
        self.last_compute = 0;
        self.previous_inputs = inputs;
        self.previous_outputs = (model.output_function)(&self.state);
        return self.previous_outputs;
    }
}

impl<const INPUTS: usize, const OUTPUTS: usize, const STATES: usize, const IRM: usize>
    Model<INPUTS, OUTPUTS, STATES, IRM>
{
    pub fn new(
        delta_time: f64,
        derivative: Box<dyn Fn(&[f64; INPUTS], &[f64; STATES]) -> [f64; STATES]>,
        output_function: Box<dyn Fn(&[f64; STATES]) -> [f64; OUTPUTS]>,
        max_input_error: [f64; INPUTS],
        max_output_error: [f64; OUTPUTS],
        // TODO: Max error states (for interpolation).
        mut scheduler_logic_size: Vec<u32>,
        scheduler_learning_rate: f64,
        scheduler_momentum: f64,
    ) -> Arc<RwLock<Model<INPUTS, OUTPUTS, STATES, IRM>>> {
        scheduler_logic_size.insert(0, (INPUTS + STATES) as u32);
        scheduler_logic_size.push(Self::all_classes().len() as u32);
        return Arc::new(RwLock::new(Model {
            delta_time,
            derivative,
            output_function,
            interpolation_data: vec![],
            advance_data: vec![],
            max_input_error,
            max_output_error,
            scheduler_data: crate::nn::NN::new(
                &scheduler_logic_size,
                scheduler_learning_rate,
                scheduler_momentum,
            ),
            samples: Mutex::new(vec![]),
            sample_fraction: 1.0,
            scheduler_enabled: false,
            tick_too_long: false,
            // TODO: Consider how to manage this number.
            //      Will it be the same for all kinetic models?
            //          -> then move to top level model.
            //      Should it be a genetic parameter?
            //      Can it be hard-coded? Const or static.
            diagnostic_period: 1000.0, // Units are ticks.
            mean_sleep: 1.0,
            mean_undersleep_factor: 1.0,
            oversleep_error_histogram: Default::default(),
        }));
    }

    /// Integrate the given state for delta_time.
    ///
    /// This deals with the IntegrationTimestep.
    fn integrate(&self, inputs: &[f64; INPUTS], state: &mut [f64; STATES]) {
        let mut t = 0.0;
        let timestep_method = IntegrationTimestep::Variable {
            error_tolerance: 1e-6,
        };
        match timestep_method {
            IntegrationTimestep::Constant(dt) => {
                while t < self.delta_time {
                    let dt = dt.min(self.delta_time - t);
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
                let mut dt = self.delta_time / 1000.0; // Initial guess for integration time step.
                let mut low_res = None;
                while t < self.delta_time {
                    dt = dt.min(self.delta_time - t);
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
                    if error <= error_tolerance * (dt / self.delta_time) {
                        *state = high_res;
                        t += dt;
                        dt *= 2.0;
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
        let mut deriv = (self.derivative)(inputs, &state);
        // Backward Euler & Crank Nicholson methods calculate the implicit
        // derivative at a future state and override the explicit derivative.
        let integration_method = IntegrationMethod::CrankNicholson { iterations: 1 };
        match &integration_method {
            IntegrationMethod::ForwardEuler => {}
            IntegrationMethod::BackwardEuler { iterations } => {
                debug_assert!(*iterations > 0);
                for _ in 0..*iterations {
                    let mut end_state = state.clone();
                    Self::ax_plus_b(&deriv, dt, &mut end_state);
                    deriv = (self.derivative)(inputs, &end_state);
                }
            }
            IntegrationMethod::CrankNicholson { iterations } => {
                debug_assert!(*iterations > 0);
                for _ in 0..*iterations {
                    let mut halfway = state.clone();
                    Self::ax_plus_b(&deriv, dt / 2.0, &mut halfway);
                    deriv = (self.derivative)(inputs, &halfway);
                }
            }
        }
        // Integrate.
        Self::ax_plus_b(&deriv, dt, state);
        // TODO: Optionally conserve the substance. Add a user defined function
        // which transforms the state in arbitrary ways.
    }

    fn ax_plus_b(a: &[f64; STATES], x: f64, b: &mut [f64; STATES]) {
        for (a_element, b_element) in a.iter().zip(b.iter_mut()) {
            *b_element += a_element * x;
        }
    }

    fn new_stm(&self, tick_power: i32, inputs: &[f64; INPUTS]) -> [f64; IRM] {
        let mut matrix = [0.; IRM];
        for src in 0..STATES {
            let mut probe = [0_f64; STATES];
            probe[src] = 1_f64;
            self.integrate(inputs, &mut probe);
            matrix[src * STATES..(src * STATES) + STATES].copy_from_slice(&probe);
        }
        for _ in 0..tick_power {
            self.square_stm(&mut matrix)
        }
        matrix
    }

    fn square_stm(&self, m: &mut [f64; IRM]) {
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

    fn apply_stm(&self, s: &[f64; STATES], m: &[f64; IRM]) -> [f64; STATES] {
        let mut result = [0f64; STATES];
        for src in 0..STATES {
            for dst in 0..STATES {
                result[dst] += s[src] * m[src * STATES + dst];
            }
        }
        result
    }

    // TODO: This function can fail if the interpolation strategy needs initialization.
    //       Make it return an Error type when that happens.

    // Previously, I waited until I had all of the failures and the dealt with
    // them as a single batch (process_failures). However, now the user is
    // calling this from the instance, and expects a synchronous result ...
    // mutex block on the main model to build the interp & advance tables.

    fn advance(&self, args: &mut AdvanceArguments<INPUTS, STATES>) {
        for table_idx in 0..32 {
            if args.elapsed_ticks & (1 << table_idx) == 0 {
                continue;
            }
            let mut next_state = [0f64; STATES];
            for (idx, weight) in self.interpolation_data[table_idx].get(args.inputs).iter() {
                // TODO: Bounds check on advance_data table access.
                let temp = self.apply_stm(&args.state, &self.advance_data[table_idx][*idx]);
                for s in 0..STATES {
                    next_state[s] += temp[s] * weight;
                }
            }
            args.state = next_state;
            let sum: f64 = args.state.iter().sum();
            for s in args.state.iter_mut() {
                *s /= sum;
            }
        }
    }

    /*
    fn process_failures(&mut self, failed_args: &[AdvanceArguments]) {
        let mut clean = vec![true; self.advance_data.len()];
        for args in failed_args {
            debug_assert!(args.inputs.iter().all(|v| v.is_finite()));
            let max_table_idx = args.max_table_idx().unwrap();
            if max_table_idx >= self.advance_data.len() {
                self.advance_data.resize_with(max_table_idx + 1, || InterpolationTable {
                    input_min: args.inputs.clone(),
                    input_max: args.inputs.clone(),
                    input_num: [2; INPUTS],
                    input_scale: Default::default(),
                    data: vec![],
                });
                clean.resize(max_table_idx + 1, false)
            }
            for table_idx in 0..=max_table_idx {
                let table = &mut self.advance_data[table_idx];
                for i in 0..INPUTS {
                    if args.inputs[i] < table.input_min[i] {
                        table.input_min[i] = args.inputs[i];
                        clean[table_idx] = false
                    }
                    if args.inputs[i] > table.input_max[i] {
                        table.input_max[i] = args.inputs[i];
                        clean[table_idx] = false
                    }
                }
            }
        }
        for (table_idx, table) in self.advance_data.iter_mut().enumerate()
                .filter(|(table_idx, _table)| !clean[*table_idx]) {
            // Increase the number of interpolation points in proportion to
            // the increase in the input data range, so as not the hurt the
            // interpolation accuracy.
            for input_idx in 0..INPUTS {
                if table.input_num[input_idx] > 2 {
                    let range = table.input_max[input_idx] - table.input_min[input_idx];
                    let num = (table.input_scale[input_idx] * range + 1.0).round();
                    debug_assert!(num >= table.input_num[input_idx] as f64);
                    table.input_num[input_idx] = num as usize;
                }
            }
            table.initialize(table_idx.try_into().unwrap());
        }
    }
    */

    /*
    fn increase_interpolation_accuracy(&mut self,
            sample: &ComputeSample) {
        let ComputeSample {args} = sample;
        let max_table_idx = match args.max_table_idx() {
            None => return,
            Some(x) => x,
        };
        let AdvanceArguments {state, inputs, elapsed_ticks} = args;
        let mut exact_stm = new_stm(-2, inputs);
        square_stm(&mut exact_stm);
        for table_idx in 0..=max_table_idx {
            square_stm(&mut exact_stm);
            let exact_state = apply_stm(state, &exact_stm);
            let exact_outputs = output_function(&exact_state);
            let ticks = 2_u32.pow(table_idx.try_into().unwrap());
            let elapsed_seconds = self.delta_time * f64::from(ticks);
            let error_frac = f64::from(ticks) / f64::from(*elapsed_ticks);
            loop {
                let mut approx_args = AdvanceArguments {
                    state: *state,
                    inputs: *inputs,
                    elapsed_ticks: ticks
                };
                approx_args.advance(&self.advance_data);
                let approx_outputs = output_function(&approx_args.state);
                let error = approx_outputs.iter().zip(&exact_outputs)
                    .map(|(approx, exact)| (approx - exact).abs() * elapsed_seconds);
                if error.zip(&self.max_output_error).any(|(err, max)| err > error_frac * *max) {

        // Notes on quantifing improvement in error:
        //      This will seek out the largest relative improvement:
        //          max((OldErr[o] - NewErr[o]) / max_err[o] for o in outputs)
        //      Alternatively, find which output has the largest relative error
        //      and optimize for only that output:
        //          target = argmax(OldErr / max_err for each output)
        //          (OldErr[target] - NewErr[target]) / max_err[target]
                    for i in 0..INPUTS {
                        self.advance_data[table_idx].input_num[i] += 1;
                    }
                    self.advance_data[table_idx].initialize(&self.parameters,
                        table_idx.try_into().unwrap());
                }
                else {
                    break
                }
            }
        }
    }
    */

    /*
    fn update_mean_sleep(&mut self, samples: &Vec<AdvanceArguments>, online_period: u64) {
        let samples = samples
            .iter()
            .map(|s| s.args.elapsed_ticks)
            .filter(|t| *t > 0);
        let mut sum = 0;
        let mut count = 0;
        for ticks in samples.clone() {
            sum += ticks as u64;
            count += 1;
        }
        let decay = (-(online_period as f64) / self.diagnostic_period).exp();
        if count > 0 {
            self.mean_sleep *= decay;
            self.mean_sleep += (1.0 - decay) * (sum as f64 / count as f64);
        }
        debug_assert!(self.mean_sleep.is_finite());
    }
    */

    /// Runs a high accuracy simulation of a ComputeSample.
    ///
    /// Return pair of (sleep, error)
    ///
    /// + Where sleep is the maximum number of ticks that the sample could
    /// have slept for
    ///
    /// + Where error is the absolute value of the integral of the
    /// difference between the assumed constant output and the true output
    /// over the length of time specified in the given sample
    /// (`sample.args.elapsed_ticks`). The error then normalized into units
    /// of max_error, and so the returned error is a multiple of max_error.
    /// An error of less than 1 is an undersleep, and an error of greater
    /// than 1 is an oversleep. There is an error associated with each
    /// output and if there are mutliple outputs then the largest error is
    /// returned.
    fn high_accuracy_advance(&mut self, sample: AdvanceArguments<INPUTS, STATES>) -> (u32, f64) {
        let mut sample_error = None; // Return value.
        let mut correct_sleep = None; // Return value.
        let max_sleep = *Self::all_classes().iter().max().unwrap();
        let mut cursor = sample;
        let sample_sleep = cursor.elapsed_ticks;
        debug_assert!(sample_sleep <= max_sleep);
        debug_assert!(sample_sleep.is_power_of_two());
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
            // if !cursor.bounds_check(&self.advance_data) {
            // self.process_failures(&[cursor]);
            // self.increase_interpolation_accuracy(&ComputeSample {
            //     args: cursor});
            todo!();
            // }
            todo!();
            // cursor.advance(&self);
            let true_output = (self.output_function)(&cursor.state);
            for o in 0..OUTPUTS {
                let trapazoid_rule = (true_output[o] + prev_output[o]) / 2.;
                error_accumulator[o] += (const_output[o] - trapazoid_rule)
                    * cursor.elapsed_ticks as f64
                    * self.delta_time;
            }
            prev_output = true_output;
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
            if sleep == sample_sleep {
                sample_error = Some(
                    error_accumulator
                        .iter()
                        .zip(self.max_output_error.iter())
                        .map(|(err, max)| err.abs() / max)
                        .fold(-f64::INFINITY, |m, err| m.max(err.abs())),
                )
            }
        }
        let correct_sleep = correct_sleep.unwrap();
        if correct_sleep == 0 {
            self.tick_too_long = true
        }
        (correct_sleep, sample_error.unwrap())
    }

    fn all_classes() -> Vec<u32> {
        // Sleep in powers of 2 because powers of 2 are fastest to compute.
        // This also greatly simplifies the schedulers problem space.
        // Further reduce the problem space by sleeping in powers of 8
        // instead of 2 after the first 10 sleep time bins.
        (0..21)
            .filter(|&x| x <= 9 || x % 3 == 0)
            .map(|x| 2_u32.pow(x))
            .collect()
    }

    fn sleep_time(active_classes: &Vec<f64>) -> u32 {
        // Shorter sleep times take precidence over longer sleep times.
        for (class, sleep) in active_classes.iter().zip(Self::all_classes()) {
            if *class > 0.5 {
                return sleep;
            }
        }
        // If the scheduler fails to classify the input as any of the sleep
        // times then assume the shortest sleep time.
        1
    }

    fn correct_classes(sleep_ticks: u32) -> Vec<usize> {
        let all_classes = Self::all_classes();
        // TODO: Move this into a unit tests!
        debug_assert!(all_classes.is_sorted());
        debug_assert!(all_classes.iter().all(|t| t.is_power_of_two()));
        let class = if sleep_ticks < all_classes[0] {
            0
        } else if sleep_ticks >= all_classes[all_classes.len() - 1] {
            all_classes.len() - 1
        } else {
            all_classes
                .iter()
                .zip(all_classes.iter().skip(1))
                .position(|range| sleep_ticks >= *range.0 && sleep_ticks < *range.1)
                .unwrap()
        };
        // Some sleep times are also classified by a few longer sleep times,
        // so that when they fail they have another classifier to fall back
        // on.
        let mut correct = vec![class];
        for redundancy in class + 1..all_classes.len() {
            if all_classes[redundancy] - all_classes[class] <= 20 {
                correct.push(redundancy);
            } else {
                break;
            }
        }
        correct
    }

    fn train_scheduler(
        &mut self,
        samples: &[AdvanceArguments<INPUTS, STATES>],
        online_period: u64,
    ) {
        let decay = (-(online_period as f64) / self.diagnostic_period).exp();
        self.oversleep_error_histogram.decay(decay);
        let mut undersleep_factor_sum: u64 = 0;
        let mut undersleep_factor_count: u64 = 0;
        for sample in samples {
            let mut inputs = Vec::with_capacity(STATES + INPUTS);
            inputs.extend_from_slice(&sample.state);
            inputs.extend_from_slice(&sample.inputs);
            let results = self.scheduler_data.run(&inputs);
            let scheduled_sleep = Self::sleep_time(&results[results.len() - 1]);
            let mut sample = *sample;
            sample.elapsed_ticks = scheduled_sleep;
            let (correct_sleep, error) = self.high_accuracy_advance(sample);
            self.oversleep_error_histogram.add(error);
            let correct_sleep_pow2 = if correct_sleep == 0 {
                correct_sleep
            } else {
                2_u32.pow(32 - 1 - correct_sleep.leading_zeros())
            };
            let undersleep_factor = correct_sleep_pow2 / scheduled_sleep;
            if undersleep_factor >= 1 {
                undersleep_factor_sum += undersleep_factor as u64;
                undersleep_factor_count += 1;
            }
            let sample_weight = error.max(1.0);
            let target_classes = Self::correct_classes(correct_sleep);
            let all_classes = Self::all_classes();
            let mut targets = vec![0.0; all_classes.len()];
            for class in target_classes.iter() {
                targets[*class] = 1.0;
            }
            self.scheduler_data.train(&results, &targets, sample_weight);
        }
        if undersleep_factor_count > 0 {
            self.mean_undersleep_factor *= decay;
            self.mean_undersleep_factor +=
                (1.0 - decay) * (undersleep_factor_sum as f64 / undersleep_factor_count as f64);
        }
        debug_assert!(self.mean_undersleep_factor.is_finite());
    }

    fn offline(&mut self, online_period: u64) {
        let samples: Vec<AdvanceArguments<INPUTS, STATES>> =
            std::mem::take(&mut self.samples.lock().unwrap());
        // self.update_mean_sleep(&samples, online_period);
        for aa in &samples {
            // self.increase_interpolation_accuracy(aa);
        }
        self.train_scheduler(&samples, online_period);
        self.scheduler_enabled = self.oversleep_error_histogram.within_tolerance();
    }

    /*
    fn diagnostics(&self) -> String {
        let mut s = "".to_string();
        if self.instances.is_empty() {
            return "No Instances.".to_string();
        }
        s += &format!("Num Instances: {}\n", self.instances.len());
        s += "Advance Data:\n";
        s += "  Ticks\tNum Tables\tMinimum\tMaximum\n";
        for (table_idx, table) in self.advance_data.iter().enumerate() {
            s += &format!("  {}\t{}\t{:?}\t{:?}\n",
                2_usize.pow(table_idx as u32),
                table.input_num.iter().product::<usize>(),
                table.input_min,
                table.input_max);
        }
        if self.scheduler_enabled {
            s += "Scheduler is ON.\n"
        }
        else {
            s += "Scheduler is OFF.\n"
        }
        // TODO: Display molecular quantity min/mean/std/max? I'm not sure about this one...
        // TODO: Display sample fraction!
        s += &format!("Average Sleep: {} Ticks.\n", self.mean_sleep);
        s += &format!("Average Under Sleep Factor {}\n", self.mean_undersleep_factor);
        s += &self.oversleep_error_histogram.to_string();
        if self.tick_too_long {
            s += "Tick Period Too Long!\n";
        }
        s
    }
    */
}
