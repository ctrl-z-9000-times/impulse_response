/*! # Model the cable equations for a neurons dendrite.

Scenario: An electrical wire / cable of infinite length. The wire is resistive
through its length. The wire is buried in the ground and is partially insulated
from the ground, and so it has a parasitic resistance and capacitance with
ground.

Apply a voltage source to one point on the wire, and observe the resulting
voltages elsewhere in the wire. In the steady state, the voltage in the wire
should exponentially decay with respect to distance from the voltage source.
Analytically solve for the steady state solution and compare. ***/

// Cable Parameters.
const AXIAL_RESISTANCE: f64 = 1e4; // Units: Ohms / Meter
const CAPACITANCE: f64 = 1e-5; // Units: Farads / Meter
const LEAK_RESISTANCE: f64 = 1e6; // Units: Ohm * Meter

// Simulation Parameters.
const CABLE_LENGTH: f64 = 500.0; // Units: Meter
const NUM_POINTS: usize = 600;
const DELTA_TIME: f64 = 1e-3;
const ACCURACY: f64 = 1e-3;

// Convert the units of length from meters to simulation-points.
const POINT_DISTANCE: f64 = (CABLE_LENGTH) / (NUM_POINTS - 1) as f64; // Units: Meters
const R_AXIAL: f64 = AXIAL_RESISTANCE * POINT_DISTANCE; // Units: Ohms
const C: f64 = CAPACITANCE * POINT_DISTANCE; // Units: Farads
const R_LEAK: f64 = LEAK_RESISTANCE / POINT_DISTANCE; // Units: Ohms

#[test]
fn leaky_cable() {
    // Setup the simulation.
    let mut point_locations = Vec::with_capacity(NUM_POINTS);
    let mut m =
        impulse_response::Model::new(DELTA_TIME, ACCURACY, DELTA_TIME / 10_000.0, f64::EPSILON);
    for idx in 0..NUM_POINTS {
        let fraction = idx as f64 / (NUM_POINTS - 1) as f64;
        point_locations.push((fraction - 0.5) * CABLE_LENGTH);
        m.touch(idx);
    }
    let center_point_location = point_locations[NUM_POINTS / 2];
    for x in point_locations.iter_mut() {
        *x -= center_point_location;
    }
    let derivative_function =
        |voltage: &impulse_response::SparseVector,
         derivative: &mut impulse_response::SparseVector| {
            for (point, v) in voltage.iter() {
                let mut total_current = v / R_LEAK;
                let mut adjacent = Vec::with_capacity(2);
                if *point > 0 {
                    adjacent.push(*point - 1)
                }
                if *point + 1 < NUM_POINTS {
                    adjacent.push(*point + 1)
                }
                for adj in adjacent {
                    match voltage.get(&adj) {
                        Some(adj_v) => {
                            total_current += (v - adj_v) / R_AXIAL;
                        }
                        None => {
                            let current = v / R_AXIAL;
                            total_current += current;
                            derivative.insert(adj, current / C);
                        }
                    }
                }
                derivative.insert(*point, -total_current / C);
            }
        };
    // Analyze the steady state response to a voltage source.
    let v_source = 3.3;
    let length_constant = f64::sqrt(LEAK_RESISTANCE / AXIAL_RESISTANCE);
    let time_constant = LEAK_RESISTANCE * CAPACITANCE;
    let steady_state =
        |point: usize| v_source * f64::exp(-point_locations[point].abs() / length_constant);
    assert!(CABLE_LENGTH > 10.0 * length_constant);
    assert!(POINT_DISTANCE < length_constant / 10.0);
    assert!(DELTA_TIME < time_constant / 10.0);
    // Measure the response using numeric integration.
    let mut voltages = vec![0.0; NUM_POINTS];
    voltages[NUM_POINTS / 2] = v_source;
    let mut next_voltages = vec![0.0; NUM_POINTS];
    let mut elapsed_time = 0.0;
    while elapsed_time <= time_constant * 20.0 {
        m.advance(&voltages, &mut next_voltages, derivative_function);
        std::mem::swap(&mut voltages, &mut next_voltages);
        voltages[NUM_POINTS / 2] = v_source;
        elapsed_time += DELTA_TIME;
    }
    // Compare the simulated voltages to the calculated voltages.
    for point in 0..NUM_POINTS {
        let exact = steady_state(point);
        let apprx = voltages[point];
        if apprx >= ACCURACY {
            // dbg!(point_locations[point], exact, apprx);
            assert!(apprx <= exact * 1.01);
            assert!(apprx >= exact * 0.99);
        }
    }
}
