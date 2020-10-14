/*** XKCD: Nerd Sniping */

// The grid of resistors is a square with side length of SIZE. In the original
// problem, the SIZE is infinity. Increasing this should improve the accuracy of
// the approximation.
// const SIZE: usize = 633;
// Or use a smaller SIZE to make it run faster.
const SIZE: usize = 32;

// Convert between 2-Dimensional grid coordinates and flat indexes into a 1-Dimensional vector.
fn coords_to_index(x: usize, y: usize) -> usize {
    x * SIZE + y
}
fn index_to_coords(idx: usize) -> (usize, usize) {
    (idx % SIZE, idx / SIZE)
}

const RESISTANCE: f64 = 1.0; // Ohms

// Add a capacitance to every node. This is required to make the units cancel
// correctly. The value is arbitrary.
const CAPACITANCE: f64 = 1.0; // Farads

// The simulation advances in increments of this time.
const TIME_STEP: f64 = RESISTANCE * CAPACITANCE / 10.0; // Seconds

// The target accuracy of the impulse response measurement.
const ACCURACY: f64 = 1e-8; // Volts

// Define the derivative of the state.
// The state of the system is the voltage at each node.
//

// Both arguments are sparse vectors. Sparse vectors are hash maps containing
// all of the non-zero values in the vector. Values which are not present are
// implicitly zeros.
fn derivative_function(
    voltage: &impulse_response::SparseVector,
    derivative: &mut impulse_response::SparseVector,
) {
    let mut adjacent = Vec::with_capacity(4);
    for (&point_index, &point_voltage) in voltage.iter() {
        // Find all adjacent nodes.
        adjacent.clear();
        let (x, y) = index_to_coords(point_index);
        if x > 0 {
            adjacent.push(coords_to_index(x - 1, y));
        }
        if x + 1 < SIZE {
            adjacent.push(coords_to_index(x + 1, y));
        }
        if y > 0 {
            adjacent.push(coords_to_index(x, y - 1));
        }
        if y + 1 < SIZE {
            adjacent.push(coords_to_index(x, y + 1));
        }
        // Measure the current into / out of this point.
        let mut total_current = 0.0;
        for adj in adjacent.iter() {
            match voltage.get(adj) {
                Some(&adj_voltage) => {
                    total_current += (point_voltage - adj_voltage) / RESISTANCE;
                }
                None => {
                    let current = point_voltage / RESISTANCE;
                    // If the adjacent point is currently excluded from the model
                    // (by virtue of being zero) and the effect on the voltage is
                    // insignificant then model the connection as an open circuit.
                    // This conserves electrical charge. The error induced by this
                    // optimization (excluding insignificant voltages from the
                    // model) self-corrects by accumulating charge in the adjacent
                    // nodes until it overcomes the voltage cutoff threshold.
                    if current / CAPACITANCE * TIME_STEP >= f64::EPSILON {
                        total_current += current;
                        derivative.insert(
                            *adj,
                            current / CAPACITANCE + derivative.get(adj).unwrap_or(&0.0),
                        );
                    }
                }
            }
        }
        derivative.insert(point_index, -total_current / CAPACITANCE);
    }
}

fn main() {
    let mut model =
        impulse_response::Model::new(TIME_STEP, ACCURACY, TIME_STEP / 10_000.0, f64::EPSILON);
    // Notify the model when ever a node is initialized or its derivative function changes.
    for node in 0..SIZE * SIZE {
        model.touch(node)
    }
    println!("Model Size: {} Nodes", model.len());
    // The model can not operate in-place, so allocate two state vectors and
    // swap them after each simulation time-step.
    let mut state = vec![0.0; SIZE * SIZE];
    let mut next_state = vec![0.0; SIZE * SIZE];
    // Coordinates of the two marked points.
    const PROBE1: (usize, usize) = (SIZE / 2, SIZE / 2);
    const PROBE2: (usize, usize) = (PROBE1.0 + 2, PROBE1.1 + 1);
    // Apply voltage source to probe1 and apply ground to probe2.
    const V_SOURCE: f64 = 1.0;
    const V_GROUND: f64 = 0.0;
    state[coords_to_index(PROBE1.0, PROBE1.1)] = V_SOURCE;
    state[coords_to_index(PROBE2.0, PROBE2.1)] = V_GROUND;
    let mut resistance = f64::NAN; // Measured resistance, Ohms
    loop {
        // Run the model.
        model.advance(&state, &mut next_state, derivative_function);
        std::mem::swap(&mut state, &mut next_state);
        // Measure the equivalent resistance.
        let delta_voltage = V_SOURCE - state[coords_to_index(PROBE1.0, PROBE1.1)];
        let current = delta_voltage * CAPACITANCE / TIME_STEP;
        let new_resistance = V_SOURCE / current;
        // Apply the voltage sources to the probe points.
        state[coords_to_index(PROBE1.0, PROBE1.1)] = V_SOURCE;
        state[coords_to_index(PROBE2.0, PROBE2.1)] = V_GROUND;
        // Run until the resistance stops changing.
        let delta_resistance = new_resistance - resistance;
        resistance = new_resistance;
        // Uncomment this debug printout to watch it converge on the steady state.
        // dbg!(resistance);
        if delta_resistance.abs() <= ACCURACY {
            break;
        }
    }
    println!("Equivalent Resistance: {} Ohms", resistance);
    println!(
        "Exact Answer: 4/PI - 1/2 = {} Ohms",
        4.0 / std::f64::consts::PI - 0.5
    );
}
