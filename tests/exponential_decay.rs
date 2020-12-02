/*! # Simple exponential decay.

Scenario: Make many points. The state value at each point exponentially
decays over time. Points are totally isolated from each other.

Analytically solve for the exact value of each point at every moment in time,
and check the accuracy of the numeric integration. ***/

use rand::prelude::*;

#[test]
fn exponential_decay() {
    // Setup the scenario.
    let num_points = 100;
    let time_constant: Vec<f64> = (0..num_points)
        .map(|_| random::<f64>() * 100.0 + 1e-6) // Tau, the time constant of exponential decay.
        .collect();
    let inital_state: Vec<f64> = (0..num_points)
        .map(|_| (random::<f64>() * 2.0 - 1.0) * 1000.0)
        .collect();
    let derivative_function =
        |state: &impulse_response::SparseVector,
         derivative: &mut impulse_response::SparseVector| {
            for (idx, value) in state.iter() {
                derivative.insert(*idx, -value / time_constant[*idx]);
            }
        };
    let simulation_duration = 20.0;
    // Setup the numeric integration.
    let delta_time = 1e-3;
    let min_dt = delta_time / 10_000.0;
    let accuracy = 1e-3;
    let mut m =
        impulse_response::Model::new(delta_time, accuracy / simulation_duration, min_dt, 0.0);
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
