/***
Conservation in a closed system.

Scenario: An element, such as heat or electric charge, diffuses throughout an
isolated system. The element flows at a rate determined by its gradient.
Eventually it will reach a steady state, and the element should be homogenously
distributed throughout the system.

An artifical system is generated:
+ It is a random directed graph, and the only constraint on the connectivity
between points is that all points have the same number of outgoing connections.
+ Each point has a random capactity to contain the element, and each edge has a
random resistance to movement through it. Points state with a random quantity of
the element.
+ Points are added and removed while the simulation is running.

Run the simulation until it reaches a steady state. Verify that the quantity of
the element is has not changed over the course of the simulation. Also, double
check the numeric integration results against the Crank-Nicholson method of
numeric integration.

***/
// #![feature(test)]
// extern crate test;
use impulse_response::{SparseModel, SparseVector};
use rand::prelude::*;
// use test::Bencher;

// Scenario parameters.
const NUM_POINTS: usize = 1_000;
const NUM_EDGES: usize = 3;

// Integration parameters.
const ACCURACY: f64 = 1e-9;
const CUTOFF: f64 = 1e-21;
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
            resistances[e] = random::<f64>() * 1e9 + CUTOFF;
        }
        Point {
            capacity: random::<f64>() * 1e5 + CUTOFF,
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
                } else if flow / points[*dst_idx].capacity * DELTA_TIME >= CUTOFF {
                    deriv.data[*src_idx] -= flow;
                    deriv.data[*dst_idx] += flow;
                    deriv.nonzero.push(*dst_idx);
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
    let mut m = SparseModel::new(DELTA_TIME, ACCURACY);
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
        let max_pct_diff = crank_nicholson
            .data
            .iter()
            .zip(&state)
            .map(|(a, b)| 2.0 * f64::abs(a - b) / (a + b))
            .fold(-1.0 / 0.0, f64::max);
        dbg!(max_pct_diff);
        assert!(max_pct_diff <= ACCURACY);
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

/*
#[bench]
fn benchmark(b: &mut Bencher) {
    b.iter(|| {
        (0..1000).fold(0, |old, new| old ^ new);
    });
}
*/
