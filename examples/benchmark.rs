/***
An artificial system is generated:
+ It is a random directed graph, and the only constraint on the connectivity
between points is that all points have the same number of outgoing connections.
+ The element flows at a rate determined by its gradient.
+ Each point has a random capacity to contain the element, and each edge has a
random resistance to movement through it. Points start with a random quantity of
the element.
+ Points are added and removed while the simulation is running.
***/

#![feature(test)]
use impulse_response::sparse::Vector as SparseVector;
use rand::prelude::*;

// Scenario parameters.
const NUM_EDGES: usize = 4;

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
    fn new(num_points: usize) -> Self {
        let mut adjacent = [0; NUM_EDGES];
        let mut resistances = [0.0; NUM_EDGES];
        for e in 0..NUM_EDGES {
            adjacent[e] = random::<usize>() % num_points;
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

fn random_graph(num_points: usize) {
    let mut points = Vec::with_capacity(num_points);
    let mut m = impulse_response::sparse::Model::new(DELTA_TIME, ACCURACY);
    for i in 0..num_points {
        points.push(Point::new(num_points));
        m.touch(i);
    }
    let mut state = Vec::with_capacity(num_points);
    for _ in 0..num_points {
        state.push(Point::random_state());
    }
    let mut next_state = vec![0.0; num_points];
    for _ in 0..num_points {
        for _ in 0..(random::<usize>() % 10) {
            // Replace a random point.
            let replace: usize = random::<usize>() % num_points;
            points[replace] = Point::new(num_points);
            state[replace] = Point::random_state() * 1000.0;
            m.touch(replace);
        }
        // Advance.
        let derivative = |s: &SparseVector, d: &mut SparseVector| Point::derivative(s, d, &points);
        m.advance(&state, &mut next_state, derivative);
        std::mem::swap(&mut state, &mut next_state);
    }
    // Advance.
    for _ in 0..num_points {
        let derivative = |s: &SparseVector, d: &mut SparseVector| Point::derivative(s, d, &points);
        m.advance(&state, &mut next_state, derivative);
        std::mem::swap(&mut state, &mut next_state);
    }
}

fn main() {
    // TODO: time each problem size independently instead of all at once.
    let start = std::time::Instant::now();
    for _ in 0..20 {
        random_graph(100);
    }
    for _ in 0..10 {
        random_graph(1000);
    }
    random_graph(10000);
    println!("Elapsed Time: {} seconds.", start.elapsed().as_secs_f64());
}
