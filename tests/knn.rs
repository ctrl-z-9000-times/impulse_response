/*! Test the Implementation of the K-Nearest-Neighbors Algorithm */

#![feature(const_generics)]
#![feature(slice_partition_at_index)]
#![allow(incomplete_features)]

use impulse_response::kd_tree::KDTree;

fn test_random_points<const DIMS: usize>(num_points: usize, k: usize) {
    let distance = |a: &[f64; DIMS], b: &[f64; DIMS]| -> f64 {
        a.iter()
            .zip(b.iter())
            // .map(|(x, y)| f64::abs(x - y))
            .map(|(x, y)| f64::powf(x - y, 2.0))
            .sum()
    };
    let mut kd_tree = KDTree::<DIMS>::default();
    let random_point = || {
        let mut point = [0.0; DIMS];
        for x in 0..DIMS {
            point[x] = rand::random::<f64>();
        }
        return point;
    };
    for _ in 0..num_points {
        // Query a random point.
        let query = random_point();
        // Solve KNN with brute force.
        let dist: Vec<_> = kd_tree
            .coordinates
            .iter()
            .map(|pt| distance(pt, &query))
            .collect();
        let cmp_dist = |a: &usize, b: &usize| dist[*a].partial_cmp(&dist[*b]).unwrap();
        let mut nn: Vec<_> = (0..kd_tree.coordinates.len()).collect();
        if nn.len() > k {
            nn.partition_at_index_by(k, cmp_dist);
            nn.truncate(k);
        }
        nn.sort_by(cmp_dist);
        // Solve KNN with the KD Tree.
        let results = kd_tree.nearest_neighbors(k, &query, &distance, f64::INFINITY);
        // Compare results.
        let payload: Vec<usize> = results.iter().map(|(x, _dist)| (*x)).collect();
        assert_eq!(payload, nn);
        // Maybe add the query point to the kd-tree.
        if rand::random() {
            kd_tree.touch(kd_tree.coordinates.len(), &query);
            kd_tree.update();
        }
    }
}

#[test]
fn knn_random_points() {
    for k in [1, 2, 3, 20].iter().copied() {
        test_random_points::<1>(600, k);
        test_random_points::<2>(600, k);
        test_random_points::<12>(600, k);
    }
}
