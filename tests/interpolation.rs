/***
Test the interpolation algorithms.

Model a complex 2-dimensional pattern with a closed form solution and check
results, to verify that the interpolation strategy works. ***/

use impulse_response::knn::KDTree;

fn test_pattern(x: f64, y: f64) -> f64 {
    use std::f64::consts::TAU;
    f64::sin(x * TAU) + f64::sin(y / 2.0 * TAU)
}

fn test_point() -> (f64, f64) {
    let size = 2.0;
    (rand::random::<f64>() * size, rand::random::<f64>() * size)
}

#[test]
fn interpolate_2d() {
    let accuracy = 0.02;
    let mut knn = KDTree::<2, 1>::default();
    let mut sample_fraction = 1.0;
    let sample_period: f64 = 1e3;
    let magic_increment: f64 = 1.7182818284590458; // Python: `1 / sum(exp(-x) for x in range(1, 100))`
    let mut successful = 0;
    while successful < 10_000 {
        let (x, y) = test_point();
        let exact_value = test_pattern(x, y);
        // Approximate using k-nearest-neighbors and interpolation.
        let approx_value = match knn.interpolate(&[x, y]) {
            Ok([approx_value]) => {
                // Continually sample the interpolation quality and respond as needed.
                if rand::random::<f64>() < sample_fraction {
                    if f64::abs(exact_value - approx_value) > accuracy / 2.0 {
                        sample_fraction += magic_increment;
                        knn.add_point(&[x, y], &[exact_value]);
                    }
                    exact_value
                } else {
                    successful += 1;
                    approx_value
                }
            }
            Err(_) => {
                sample_fraction += magic_increment;
                knn.add_point(&[x, y], &[exact_value]);
                exact_value
            }
        };
        sample_fraction *= f64::exp(-1.0 / sample_period);
        // Check results.
        assert!(f64::abs(exact_value - approx_value) <= accuracy);
    }
    println!("Interpolation Points: {}", knn.len());
    assert!(knn.len() < 125_000);
}
