/*! Test the interpolation algorithms.

Model a complex 2-dimensional pattern with a closed form solution and check
results, to verify that the interpolation strategy works. ***/

use impulse_response::knn_interp::KnnInterpolator;

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
    let knn = KnnInterpolator::<2, 1>::new(1e3, 0.0);
    let mut successful = 0.0;
    while successful < 1e4 {
        let (x, y) = test_point();
        let exact_value = test_pattern(x, y);
        // Approximate using k-nearest-neighbors and interpolation.
        let approx_value = knn.interpolate(
            &[x, y],
            |_point| [exact_value],
            |a: &[f64; 1], b: &[f64; 1]| {
                if f64::abs(a[0] - b[0]) > accuracy / 2.0 {
                    Err(())
                } else {
                    Ok(())
                }
            },
        );
        // Check results.
        assert!(f64::abs(exact_value - approx_value[0]) <= accuracy);
        successful += 1.0 - knn.sample_fraction();
    }
    println!("{}", knn);
    assert!(knn.len() < 125_000);
}
