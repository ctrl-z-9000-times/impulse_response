/*! Conservation in a closed system.

Scenario: An element moves throughout an isolated system, an element such as
heat or electric charge. The element is conserved, meaning that the element can
not be spontaneously created or destroyed. Model such a system, and verify.

An artificial system is generated:
+ It is a random directed graph, and the only constraint on the connectivity
between points is that all points have the same number of outgoing connections.
+ The element flows at a rate determined by its gradient.
+ Each point has a random capacity to contain the element, and each edge has a
random resistance to movement through it. Points start with a random quantity of
the element.
+ Points are added and removed while the simulation is running.

Run the simulation until it reaches a steady state. Verify that the simulation
has not lost track of the total quantity of the element. Also, double check the
numeric integration results against the Crank-Nicholson method of numeric
integration. ***/

use impulse_response::sparse::Vector as SparseVector;
use rand::prelude::*;

// Scenario parameters.
const NUM_POINTS: usize = if cfg!(debug_assertions) { 100 } else { 1000 };
const NUM_EDGES: usize = 3;

// Integration parameters.
const ACCURACY: f64 = 1e-9;
const DELTA_TIME: f64 = 1e-4;

/// The system is a vector of points.
struct Point {
    capacity: f64,
    adjacent: [usize; NUM_EDGES],
    resistances: [f64; NUM_EDGES],
}

impl Point {
    fn new() -> Self {
        let mut adjacent = [0; NUM_EDGES];
        let mut resistances = [0.0; NUM_EDGES];
        for e in 0..NUM_EDGES {
            adjacent[e] = random::<usize>() % NUM_POINTS;
            resistances[e] = random::<f64>() * 1e9 + ACCURACY;
        }
        Point {
            capacity: random::<f64>() * 1e5 + ACCURACY,
            adjacent,
            resistances,
        }
    }

    fn random_state() -> f64 {
        1.0 / (random::<f64>() + 1e-6)
    }

    fn derivative(state: &SparseVector, deriv: &mut SparseVector, points: &[Point]) {
        for src_idx in &state.nonzero {
            deriv.nonzero.push(*src_idx);
            let src = &points[*src_idx];
            for (dst_idx, resist) in src.adjacent.iter().zip(&src.resistances) {
                let dst_state = &state.data[*dst_idx];
                let flow = (state.data[*src_idx] - dst_state) / resist;
                let flow = flow.max(0.0); // One way connections.
                if *dst_state != 0.0 {
                    deriv.data[*src_idx] -= flow;
                    deriv.data[*dst_idx] += flow;
                } else {
                    if flow >= f64::EPSILON {
                        deriv.data[*src_idx] -= flow;
                        deriv.data[*dst_idx] += flow;
                        deriv.nonzero.push(*dst_idx);
                    }
                }
            }
        }
        for p in &deriv.nonzero {
            deriv.data[*p] /= points[*p].capacity;
        }
    }
}

#[test]
fn conservation() {
    let mut points = Vec::with_capacity(NUM_POINTS);
    // Do not apply any cutoff BC it alters the results, causing this to deviate
    // from the basic numeric integration method results.
    let mut m = impulse_response::sparse::Model::new(DELTA_TIME, ACCURACY, 0.0);
    for i in 0..NUM_POINTS {
        points.push(Point::new());
        m.touch(i);
    }
    let mut state = Vec::with_capacity(NUM_POINTS);
    for _ in 0..NUM_POINTS {
        state.push(Point::random_state());
    }
    let mut next_state = vec![0.0; NUM_POINTS];
    let mut crank_nicholson = SparseVector::new(NUM_POINTS);
    crank_nicholson.data = state.clone();
    crank_nicholson.nonzero = (0..NUM_POINTS).into_iter().collect();
    let mut initial_quantity: f64 = state.iter().zip(&points).map(|(s, p)| s * p.capacity).sum();
    for _ in 0..NUM_POINTS {
        // Replace a random point.
        let replace: usize = random::<usize>() % NUM_POINTS;
        initial_quantity -= state[replace] * points[replace].capacity;
        points[replace] = Point::new();
        state[replace] = Point::random_state() * 1000.0;
        crank_nicholson.data[replace] = state[replace];
        initial_quantity += state[replace] * points[replace].capacity;
        m.touch(replace);
        // Advance the numeric integrations.
        let mut derivative =
            |s: &SparseVector, d: &mut SparseVector| Point::derivative(s, d, &points);
        m.advance(&state, &mut next_state, derivative);
        std::mem::swap(&mut state, &mut next_state);
        crank_nicholson = m.integrate(crank_nicholson, &mut derivative); // Private method.
        let crank_nicholson_error = crank_nicholson
            .data
            .iter()
            .zip(&state)
            .map(|(a, b)| 2.0 * f64::abs(a - b) / (a + b))
            .fold(-f64::INFINITY, f64::max);
        dbg!(crank_nicholson_error);
        assert!(crank_nicholson_error <= ACCURACY);
    }
    // Run the model until it stops changing.
    let mut steady_state = false;
    for _ in 0..1_000_000 {
        m.advance(
            &state,
            &mut next_state,
            |s: &SparseVector, d: &mut SparseVector| Point::derivative(s, d, &points),
        );
        std::mem::swap(&mut state, &mut next_state);
        let max_pct_diff: f64 = state
            .iter()
            .zip(&next_state)
            .map(|(a, b)| (a - b).abs() / a)
            .fold(-1.0 / 0.0, f64::max);
        dbg!(max_pct_diff);
        if max_pct_diff <= ACCURACY {
            steady_state = true;
            break;
        }
    }
    assert!(steady_state);
    // Check that it conserved the total state.
    let final_quantity__: f64 = state.iter().zip(&points).map(|(s, p)| s * p.capacity).sum();
    let abs_diff = (initial_quantity - final_quantity__).abs();
    dbg!(initial_quantity, final_quantity__);
    assert!(abs_diff / initial_quantity <= ACCURACY);
}
