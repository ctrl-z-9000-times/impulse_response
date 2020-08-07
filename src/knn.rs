/*! K-Nearest-Neighbors */

// TODO: Implement max_distance for interpolations. Simultaneously enable the
// min_k parameter. This will allow it to discard a few outlier interpolation
// points without failing.

use rand::prelude::*;
use rayon::prelude::*;

pub struct KDTree<const DIMS: usize, const PAYLOAD: usize> {
    tree: Vec<Node>,
    points: Vec<[f64; DIMS]>,
    outputs: Vec<[f64; PAYLOAD]>,
    params: Parameters<DIMS>,
}

/// These parameters improved the interpolation.
#[derive(Debug, Copy, Clone, PartialEq)]
struct Parameters<const DIMS: usize> {
    /// Number of neighbors to return to the user.
    k: usize,

    /// Minimum number of neighbors needed for interpolation.
    min_k: usize,

    /// Each input dimension has its own units.
    scale_factor: [f64; DIMS],

    /// Max distance between points before they're not neighbors.
    max_distance: f64,
}

impl<const DIMS: usize> Parameters<DIMS> {
    /// Manhattan Distance.
    fn distance(&self, a: &[f64; DIMS], b: &[f64; DIMS]) -> f64 {
        let mut sum = 0.0;
        for i in 0..DIMS {
            sum += f64::abs((a[i] - b[i]) * self.scale_factor[i]);
        }
        return sum;
    }

    fn all_mutations(&self) -> Vec<Self> {
        let mut all = vec![];
        for delta in [-3, -2, -1, 1, 2, 3].iter() {
            let mut clone = self.clone();
            if clone.k as isize + delta < 1 {
                continue;
            }
            clone.k = (clone.k as isize + delta).max(1) as usize;
            clone.min_k = (clone.min_k as isize + delta).max(1) as usize;
            all.push(clone);
        }
        // for delta in [-3, -2, -1, 1, 2, 3] {
        //     let mut clone = self.clone();
        //     clone.min_k = ((clone.min_k as isize + delta.max(&1)) as usize).min(clone.k);
        //     all.push(clone);
        // }
        for percent_change in &[
            0.1, 1.0, 10.0, 50.0, 90.0, 95.0, 98.0, 99.5, 100.5, 102.0, 105.0, 110.0, 200.0,
            1_000.0, 10_000.0, 100_000.0,
        ] {
            for dim in 0..DIMS {
                let mut clone = self.clone();
                clone.scale_factor[dim] *= percent_change / 100.0;
                all.push(clone);
            }
            // let mut clone = self.clone();
            // clone.max_distance *= percent_change / 100.0;
            // all.push(clone);
        }
        return all;
    }

    #[allow(dead_code)]
    fn recombine(&self, other: &Parameters<DIMS>) -> Parameters<DIMS> {
        let rng = &mut thread_rng();
        let mut p = Parameters {
            k: *[self.k, other.k].choose(rng).unwrap(),
            min_k: *[self.min_k, other.min_k].choose(rng).unwrap(),
            scale_factor: self.scale_factor,
            max_distance: *[self.max_distance, other.max_distance].choose(rng).unwrap(),
        };
        for i in 0..DIMS {
            if rng.gen() {
                p.scale_factor[i] = other.scale_factor[i];
            }
        }
        return p;
    }
}

#[derive(Debug, Copy, Clone)]
enum Node {
    Partition(f64),
    Leaf(usize),
}

impl Node {
    fn root() -> usize {
        0
    }

    #[allow(dead_code)]
    fn parent(index: usize) -> Option<usize> {
        if index == Self::root() {
            None // Root.
        } else {
            Some((index - 1) / 2)
        }
    }

    fn children(index: usize) -> [usize; 2] {
        return [index * 2 + 1, index * 2 + 2];
    }

    /// depth(root) -> 1.
    fn depth(index: usize) -> usize {
        (0_usize.count_zeros() - (index + 1).leading_zeros()) as usize
    }
}

impl<const DIMS: usize, const PAYLOAD: usize> Default for KDTree<DIMS, PAYLOAD> {
    fn default() -> KDTree<DIMS, PAYLOAD> {
        KDTree {
            tree: vec![],
            points: vec![],
            outputs: vec![],
            params: Parameters {
                k: DIMS + 2,
                min_k: DIMS + 2,
                scale_factor: [1.0; DIMS],
                max_distance: f64::INFINITY,
            },
        }
    }
}

impl<const DIMS: usize, const PAYLOAD: usize> KDTree<DIMS, PAYLOAD> {
    /// Number of interpolation points.
    pub fn len(&self) -> usize {
        return self.points.len();
    }

    pub fn add_point(&mut self, point: &[f64; DIMS], datum: &[f64; PAYLOAD]) {
        debug_assert!(point.iter().all(|x| x.is_finite()));
        if self.len() >= 42 && self.len() == self.len().next_power_of_two() {
            self.hill_climb();
        }
        let index = self.points.len();
        self.points.push(*point);
        self.outputs.push(*datum);
        // Special case for the root.
        if self.tree.len() == 0 {
            self.tree.push(Node::Leaf(index));
            return;
        }
        // Recursive descent through the KD Tree to find insertion point.
        let mut cursor = Node::root();
        let tree_depth = Node::depth(self.tree.len() - 1);
        loop {
            if Node::depth(cursor) + 1 >= tree_depth {
                // If inserting the new node would exceed the current memory
                // allocation, then rebuild the whole table to be more balanced.
                self.rebuild_tree();
                break;
            }
            match self.tree[cursor] {
                Node::Partition(split_plane) => {
                    let axis = Node::depth(cursor) % DIMS;
                    let [lesser, greater] = Node::children(cursor);
                    // Determine which side of the splitting plane the query
                    // point is on.
                    let (closer, _further) = if self.points[index][axis] < split_plane {
                        (lesser, greater)
                    } else {
                        (greater, lesser)
                    };
                    // Move the cursor onto the closer child and continue descent.
                    cursor = closer;
                }
                Node::Leaf(leaf) => {
                    // Replace this leaf with a partition leading to both the
                    // old leaf and the new point.
                    self.build_tree_recursive(cursor, &mut [leaf, index]);
                    break;
                }
            }
        }
    }

    fn rebuild_tree(&mut self) {
        self.tree.clear();
        self.tree.resize(
            // Allocate enough space for the tree to expand in an unbalanced
            // way, for up to 2 levels beyond the extent of the initially
            // balanced tree.
            self.points.len().next_power_of_two() * (1 + 1 + 2 + 4) - 1,
            Node::Leaf(usize::MAX), // Poison value.
        );
        let mut leaves: Vec<_> = (0..self.points.len()).collect();
        self.build_tree_recursive(Node::root(), &mut leaves);
    }

    fn build_tree_recursive(&mut self, node: usize, leaves: &mut [usize]) {
        debug_assert!(!leaves.is_empty());
        if leaves.len() == 1 {
            self.tree[node] = Node::Leaf(leaves[0]);
        } else {
            let axis = Node::depth(node) % DIMS;
            // Divide the leaves in half, split by an axis aligned plane.
            let half = leaves.len() / 2;
            let (lesser, pivot, _greater) = leaves.partition_at_index_by(half, |x, y| {
                self.points[*x][axis]
                    .partial_cmp(&self.points[*y][axis])
                    .unwrap()
            });
            // Find the largest element which is less than the pivot. Make the
            // splitting plane halfway between these two elements.
            let max_lesser = lesser
                .iter()
                .fold(-f64::INFINITY, |max, idx| max.max(self.points[*idx][axis]));
            self.tree[node] = Node::Partition((self.points[*pivot][axis] + max_lesser) / 2.0);
            // Recurse.
            let [child1, child2] = Node::children(node);
            let (lesser, greater) = leaves.split_at_mut(half);
            self.build_tree_recursive(child1, lesser);
            self.build_tree_recursive(child2, greater);
        }
    }

    /// Finds the K nearest neighbors to a point.
    ///
    /// Returns pairs of (payload, distance).
    pub fn nearest_neighbors(&self, point: &[f64; DIMS], k: usize) -> Vec<(&[f64; PAYLOAD], f64)> {
        self.nearest_neighbors_inner(
            point,
            &Parameters {
                k,
                min_k: 0,
                scale_factor: [1.0; DIMS],
                max_distance: f64::INFINITY,
            },
        )
    }

    fn nearest_neighbors_inner(
        &self,
        point: &[f64; DIMS],
        params: &Parameters<DIMS>,
    ) -> Vec<(&[f64; PAYLOAD], f64)> {
        debug_assert!(params.k > 0);
        if self.points.is_empty() {
            return vec![];
        }
        // Neighbors is priority queue, sorted by distance.
        let mut neighbors: Vec<(&[f64; PAYLOAD], f64)> = Vec::with_capacity(params.k);
        let mut max_neighbor_dist = params.max_distance;
        // Save index of node, and the coordinates of the corner of its bounding
        // box which is closest to the query point.
        struct StackData<const DIMS: usize> {
            node: usize,
            corner: [f64; DIMS],
        }
        let mut stack = Vec::with_capacity(Node::depth(self.points.len() - 1));
        stack.push(StackData {
            node: Node::root(),
            corner: *point,
        });
        loop {
            let mut cursor = match stack.pop() {
                Some(data) => data,
                None => break,
            };
            // Check if this node can be discarded because its bounding box is
            // further from the query point than the already-found neighbors.
            if params.distance(&point, &cursor.corner) >= max_neighbor_dist {
                continue;
            }
            // Descend until it reaches a leaf.
            loop {
                match self.tree[cursor.node] {
                    Node::Partition(split_plane) => {
                        let axis = Node::depth(cursor.node) % DIMS;
                        let [lesser, greater] = Node::children(cursor.node);
                        // Determine which side of the splitting plane the query
                        // point is on.
                        let (closer, further) = if point[axis] < split_plane {
                            (lesser, greater)
                        } else {
                            (greater, lesser)
                        };
                        // Push the further child onto the stack to deal with it later.
                        let mut new_corner = cursor.corner;
                        new_corner[axis] = split_plane;
                        stack.push(StackData {
                            node: further,
                            corner: new_corner,
                        });
                        // Move the cursor onto the closer child and continue descent.
                        cursor.node = closer;
                    }
                    Node::Leaf(index) => {
                        let dist = params.distance(&point, &self.points[index]);
                        if dist < max_neighbor_dist {
                            // Insert this point into the nearest neighbors priority queue.
                            if neighbors.len() == params.k {
                                neighbors.pop();
                            }
                            let sorted_position = neighbors
                                .iter()
                                .position(|(_, nn_dist)| dist < *nn_dist)
                                .unwrap_or(neighbors.len());
                            neighbors.insert(sorted_position, (&self.outputs[index], dist));
                            max_neighbor_dist = *neighbors
                                .get(params.k - 1)
                                .map(|(_, max_nn_dist)| max_nn_dist)
                                .unwrap_or(&params.max_distance);
                        }
                        break;
                    }
                }
            }
        }
        return neighbors;
    }

    /// Interpolates the payload at any given point.
    pub fn interpolate(&self, point: &[f64; DIMS]) -> Result<[f64; PAYLOAD], ()> {
        self.interpolate_inner(point, &self.params, false)
    }

    fn interpolate_inner(
        &self,
        point: &[f64; DIMS],
        params: &Parameters<DIMS>,
        discard_closest: bool,
    ) -> Result<[f64; PAYLOAD], ()> {
        let mut nearest_neighbors = self.nearest_neighbors_inner(point, params);
        let nearest_neighbors = if discard_closest {
            let len = nearest_neighbors.len();
            &mut nearest_neighbors[1..len]
        } else {
            &mut nearest_neighbors
        };
        if nearest_neighbors.len() < self.params.min_k {
            return Err(());
        }
        // Transform the distances into interpolation weights.
        for (_, distance) in nearest_neighbors.iter_mut() {
            *distance = 1.0 / distance.max(f64::EPSILON); // Do not divide by zero.
        }
        let sum: f64 = nearest_neighbors.iter().map(|(_, weight)| weight).sum();
        // Interpolate.
        let mut result = [0.0; PAYLOAD];
        for (payload, weight) in nearest_neighbors.iter() {
            for (out, value) in result.iter_mut().zip(payload.iter()) {
                *out += value * weight / sum;
            }
        }
        return Ok(result);
    }

    fn interpolation_error(&self, selected_points: &[usize], params: &Parameters<DIMS>) -> f64 {
        let mut errors = Vec::with_capacity(selected_points.len());
        // Increase K, the number of interpolation points, by one because
        // the selected point will be discarded.
        let mut params = params.clone();
        params.k += 1;
        for point_index in selected_points {
            let exact_value = self.outputs[*point_index];
            let approx_value = self.interpolate_inner(&self.points[*point_index], &params, true);
            errors.push(match approx_value {
                Ok(approx_value) => {
                    // Root Mean Square Error.
                    f64::sqrt(
                        approx_value
                            .iter()
                            .zip(exact_value.iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f64>()
                            / PAYLOAD as f64,
                    )
                }
                Err(_) => f64::NAN, // Interpolation failed.
            });
        }
        // Find the mean error by excluding any missing interpolation values.
        let (num_ok, sum_ok) = errors
            .iter()
            .filter(|x| !x.is_nan())
            .enumerate()
            .fold((0, 0.0), |a, b| (b.0, a.1 + b.1));
        let mean_error = if num_ok > 0 {
            sum_ok / num_ok as f64
        } else {
            f64::INFINITY
        };
        // Replace NaNs with 125% of the mean_error.
        let missing_data_penalty = 1.25 * mean_error * (errors.len() - num_ok) as f64;
        return (sum_ok + missing_data_penalty) / errors.len() as f64;
    }

    pub fn hill_climb(&mut self) {
        let sample_points: Vec<_> =
            (0..self.len()).choose_multiple(&mut thread_rng(), 100_000.min(self.len()));
        let mut current_error = self.interpolation_error(&sample_points, &self.params);
        for _ in 0..10 {
            let directions = self.params.clone().all_mutations();
            let errors: Vec<f64> = directions
                .par_iter()
                .map(|indiv| self.interpolation_error(&sample_points, indiv))
                .collect();
            let mut pop_index: Vec<usize> = (0..directions.len()).collect();
            pop_index.partition_at_index_by(0, |a, b| errors[*a].partial_cmp(&errors[*b]).unwrap());
            let best_error = errors[pop_index[0]];
            if best_error < current_error {
                self.params = directions[pop_index[0]];
                current_error = best_error;
            } else {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_and_check_random_points<const DIMS: usize>(k: usize) {
        let random = || {
            let mut point = [0.0; DIMS];
            for x in 0..DIMS {
                point[x] = rand::random::<f64>();
            }
            return point;
        };
        let mut knn = KDTree::<DIMS, 1>::default();
        let distance = |a: &[f64], b: &[f64]| -> f64 {
            a.iter().zip(b.iter()).map(|(x, y)| f64::abs(x - y)).sum()
        };
        for _ in 0..200 {
            // Query a random point.
            let query = random();
            // Solve KNN with brute force.
            let dist: Vec<_> = knn.points.iter().map(|pt| distance(pt, &query)).collect();
            let cmp_dist = |a: &usize, b: &usize| dist[*a].partial_cmp(&dist[*b]).unwrap();
            let mut nn: Vec<_> = (0..knn.points.len()).collect();
            if nn.len() > k {
                nn.partition_at_index_by(k, cmp_dist);
                nn.truncate(k);
            }
            nn.sort_by(cmp_dist);
            // Solve KNN with the KD Tree.
            let results = knn.nearest_neighbors(&query, k);
            // Compare results.
            let payload: Vec<usize> = results.iter().map(|(x, _)| (**x)[0] as usize).collect();
            assert_eq!(payload, nn);
            // Maybe add the query point to the kd-tree.
            if rand::random() {
                knn.add_point(&query, &[knn.points.len() as f64; 1]);
            }
        }
    }

    #[test]
    fn random_points() {
        for k in [1, 2, 3, 20].iter().copied() {
            test_and_check_random_points::<1>(k);
            test_and_check_random_points::<2>(k);
            test_and_check_random_points::<12>(k);
        }
    }
}
