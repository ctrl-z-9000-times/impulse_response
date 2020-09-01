/*! */

use impulse_response::kd_tree::KDTree;
use impulse_response::voronoi::{ConvexHull, Face};

fn cubes_in_cube(cubes_in_row: usize, jitter: f64) -> ((f64, f64), (f64, f64)) {
    let world = ConvexHull::aabb(&[0.0, 0.0, 0.0], &[1.0, 1.0, 1.0]);
    // dbg!(&world);
    let mut tree = KDTree::<3>::default();
    let radius = 1.0 / (2.0 * cubes_in_row as f64);
    let rnd = || radius * (rand::random::<f64>() * 2.0 - 1.0) * jitter;
    for x_idx in 0..cubes_in_row {
        let x_coord = radius * (1 + 2 * x_idx) as f64;
        for y_idx in 0..cubes_in_row {
            let y_coord = radius * (1 + 2 * y_idx) as f64;
            for z_idx in 0..cubes_in_row {
                let z_coord = radius * (1 + 2 * z_idx) as f64;
                tree.touch(
                    tree.len(),
                    &[x_coord + rnd(), y_coord + rnd(), z_coord + rnd()],
                );
            }
        }
    }
    tree.update();
    let mut v_stats = Vec::with_capacity(tree.len());
    let mut sa_stats = Vec::with_capacity(tree.len());
    for x in 0..tree.len() {
        let qq = ConvexHull::new(&tree, x, &world);
        v_stats.push(qq.volume);
        sa_stats.push(qq.faces.iter().map(|f| f.surface_area).sum::<f64>());
        // TODO: check that the facing locations are sane.
    }
    let len = tree.len() as f64;
    let v_mean: f64 = v_stats.iter().sum::<f64>() / len;
    let v_std = f64::sqrt(
        v_stats
            .iter()
            .map(|&v| f64::powi(v - v_mean, 2))
            .sum::<f64>()
            / len,
    );
    let sa_mean: f64 = sa_stats.iter().sum::<f64>() / len;
    let sa_std = f64::sqrt(
        sa_stats
            .iter()
            .map(|&v| f64::powi(v - sa_mean, 2))
            .sum::<f64>()
            / len,
    );
    return ((v_mean, v_std), (sa_mean, sa_std));
}

#[test]
fn tesselating_cubes() {
    fn float_eq(left: f64, right: f64, tolerance: f64) -> bool {
        let x = f64::abs(left - right) <= tolerance;
        if !x {
            dbg!(left, right);
        }
        return x;
    }

    // Test perfectly uniform grids.
    for cubes_in_row in 1..=if cfg!(debug_assertions) { 5 } else { 7 } {
        // dbg!(cubes_in_row);
        let side_length = 1.0 / cubes_in_row as f64;
        let ((v_mean, v_std), (sa_mean, sa_std)) = cubes_in_cube(cubes_in_row, 0.0);
        assert!(float_eq(v_mean, side_length.powi(3), f64::EPSILON));
        assert!(float_eq(v_std, 0.0, f64::EPSILON));
        assert!(float_eq(
            sa_mean,
            6.0 * side_length.powi(2),
            10.0 * f64::EPSILON
        ));
        assert!(float_eq(sa_std, 0.0, 10.0 * f64::EPSILON));
    }

    // Test with very slight jitter. This test case hits many floating point
    // issues.
    let num = if cfg!(debug_assertions) { 5 } else { 13 };
    let ((v_mean, v_std), (sa_mean, sa_std)) = cubes_in_cube(num, 0.00000000001);
    // dbg!(v_mean, v_std, sa_mean, sa_std);
    let atol = 0.005; // Not great...
    assert!(float_eq(v_mean * (num as f64).powi(3), 1.0, atol));
    assert!(float_eq(v_std, 0.0, atol));
    assert!(float_eq(sa_mean, 6.0 * (1.0 / num as f64).powi(2), atol));
    assert!(float_eq(sa_std, 0.0, atol));

    // Test with more jitter.
    let num = if cfg!(debug_assertions) { 5 } else { 9 };
    let ((v_mean, v_std), (sa_mean, sa_std)) = cubes_in_cube(num, 1e-4);
    // dbg!(v_mean, v_std, sa_mean, sa_std);
    assert!(float_eq(v_mean * (num as f64).powi(3), 1.0, 1e-9));
    assert!(float_eq(v_std, 0.0, 1e-5));
    assert!(float_eq(sa_mean, 6.0 * (1.0 / num as f64).powi(2), 1e-5));
    assert!(float_eq(sa_std, 0.0, 1e-4));

    // Test with large jitter.
    let num = 5;
    let ((v_mean, v_std), (sa_mean, sa_std)) = cubes_in_cube(num, 0.99);
    // dbg!(v_mean, v_std, sa_mean, sa_std);
    assert!(float_eq(v_mean * (num as f64).powi(3), 1.0, 1e-5));
    assert!(float_eq(v_std, 0.0, 0.01));
    assert!(float_eq(sa_mean, 6.0 * (1.0 / num as f64).powi(2), 0.1));
    assert!(float_eq(sa_std, 0.0, 0.1));
}

/// This test verifies some convex hull / nearest neighbor properties.
#[test]
fn nearest_neighbor() {
    const DIMS: usize = 3;
    let distance = |a: &[f64; DIMS], b: &[f64; DIMS]| -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| f64::powf(x - y, 2.0))
            .sum()
    };
    let random_point = || {
        let mut point = [0.0; DIMS];
        for x in 0..DIMS {
            point[x] = rand::random::<f64>();
        }
        return point;
    };
    let mut kd_tree = KDTree::<DIMS>::default();
    for x in 0..if cfg!(debug_assertions) { 100 } else { 1_000 } {
        kd_tree.touch(x, &random_point());
    }
    kd_tree.update();
    let world = ConvexHull::aabb(&[0.0, 0.0, 0.0], &[1.0, 1.0, 1.0]);
    let mut cells = Vec::with_capacity(kd_tree.len());
    for x in 0..kd_tree.len() {
        cells.push(ConvexHull::new(&kd_tree, x, &world));
    }
    // Demonstrate how to add more cells on-line.
    for new in kd_tree.len()..kd_tree.len() + 100 {
        kd_tree.touch(new, &random_point());
        kd_tree.update();
        let new_cell = ConvexHull::new(&kd_tree, new, &world);
        // Recompute all of the cells which the new cell touches.
        for Face {
            facing_location, ..
        } in new_cell.faces.iter()
        {
            if let Some(x) = *facing_location {
                cells[x] = ConvexHull::new(&kd_tree, x, &world)
            }
        }
        cells.push(new_cell);
    }
    // Test many random locations.
    for _y in 0..if cfg!(debug_assertions) {
        10_000
    } else {
        1_000_000
    } {
        let test_point = random_point();
        let nn = kd_tree.nearest_neighbors(1, &test_point, &distance, f64::INFINITY);
        let nn = nn[0].0;
        assert!(cells[nn].contains_point(&test_point));
        let not_nn = nn.checked_sub(1).unwrap_or(42);
        assert!(!cells[not_nn].contains_point(&test_point));
    }
}
