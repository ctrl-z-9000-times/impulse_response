/***
Simple exponential decay.

Scenario: Make many points. The state value at each point exponentially
decays over time. Points are totally isolated from each other.

Analytically solve for the exact value of each point at every moment in time,
and check the accuracy of the numeric integration.
***/
#![feature(const_int_pow)]

use rand::prelude::*;

#[test]
fn sparse() {
    // Setup the scenario.
    let num_points = 100;
    let time_constant: Vec<f64> = (0..num_points)
        .map(|_| random::<f64>() * 100.0 + 1e-6) // Tau, the time constant of exponential decay.
        .collect();
    let inital_state: Vec<f64> = (0..num_points)
        .map(|_| (random::<f64>() * 2.0 - 1.0) * 1000.0)
        .collect();
    let derivative_function =
        |state: &impulse_response::sparse::Vector,
         derivative: &mut impulse_response::sparse::Vector| {
            for idx in &state.nonzero {
                derivative.data[*idx] = -state.data[*idx] / time_constant[*idx];
                derivative.nonzero.push(*idx);
            }
        };
    let simulation_duration = 20.0;
    // Setup the numeric integration.
    let delta_time = 1e-3;
    let accuracy = 1e-6;
    let mut m = impulse_response::sparse::Model::new(
        delta_time,
        accuracy / (simulation_duration / delta_time),
        0.0,
    );
    for i in 0..num_points {
        m.touch(i);
    }
    let mut elapsed_time = 0.0;
    let mut state = inital_state.clone();
    let mut next_state = vec![0.0; num_points];
    // Run the numeric integration.
    while elapsed_time < simulation_duration {
        m.advance(&state, &mut next_state, derivative_function);
        std::mem::swap(&mut state, &mut next_state);
        elapsed_time += delta_time;
        // Compare the results of numeric integration to analytic integration.
        for i in 0..num_points {
            let exact = inital_state[i] * f64::exp(-elapsed_time / time_constant[i]);
            let abs_diff = (state[i] - exact).abs();
            if abs_diff > accuracy {
                panic!(
                    "{} e^(-dt / {}) = exact {}, approx {}",
                    inital_state[i], time_constant[i], exact, state[i]
                );
            }
        }
    }
}

fn dense_1() {
    // Setup the scenario.
    const NUM_POINTS: usize = 1;
    let time_constant: Vec<f64> = (0..NUM_POINTS)
        .map(|_| random::<f64>() * 100.0 + 1e-6) // Tau, the time constant of exponential decay.
        .collect();
    let time_constant_copy = time_constant.clone();
    let inital_state: Vec<f64> = (0..NUM_POINTS)
        .map(|_| (random::<f64>() * 2.0 - 1.0) * 1000.0)
        .collect();
    let derivative_function =
        move |_inputs: &[f64; 1], state: &[f64; NUM_POINTS]| -> [f64; NUM_POINTS] {
            let mut deriv = state.clone();
            deriv
                .iter_mut()
                .zip(time_constant_copy.iter())
                .for_each(|(s, tau)| *s = -*s / tau);
            deriv
        };
    let simulation_duration = 20.0;
    // Setup the numeric integration.
    let delta_time = 1e-2;
    let accuracy = 1e-6;
    const SQR: usize = NUM_POINTS.pow(2);
    const INPUTS_STATES: usize = 1 + NUM_POINTS;
    let model =
        impulse_response::dense::Model::<1, NUM_POINTS, NUM_POINTS, SQR, INPUTS_STATES>::new(
            delta_time,
            Box::new(derivative_function),
            Box::new(|state| *state),
            None,
            &[1.0; 1],
            &[accuracy / (simulation_duration / delta_time); NUM_POINTS],
            &[accuracy / (simulation_duration / delta_time); NUM_POINTS],
        );
    let mut inst = impulse_response::dense::Instance::new(&model, inital_state.as_slice());
    let mut elapsed_time = 0.0;
    // Run the numeric integration.
    while elapsed_time < simulation_duration {
        let state = inst.advance(Default::default());
        elapsed_time += delta_time;
        // Compare the results of numeric integration to analytic integration.
        for i in 0..NUM_POINTS {
            let exact = inital_state[i] * f64::exp(-elapsed_time / time_constant[i]);
            let abs_diff = (state[i] - exact).abs();
            if abs_diff > accuracy {
                panic!(
                    "{} e^(-dt / {}) = exact {}, approx {}",
                    inital_state[i], time_constant[i], exact, state[i]
                );
            }
        }
    }
}

#[test]
fn dense() {
    for _i in 0..100 {
        dense_1();
    }
}
