/*! K-Dimensional Tree */

#[derive(Default)]
pub struct KDTree<const DIMS: usize> {
    _touched: Vec<usize>,
    tree: Vec<Node>,
    pub coordinates: Vec<[f64; DIMS]>,
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

impl<const DIMS: usize> KDTree<DIMS> {
    /// Number of points in the tree.
    pub fn len(&self) -> usize {
        return self.coordinates.len();
    }

    pub fn touch(&mut self, location: usize, coordinates: &[f64; DIMS]) {
        debug_assert!(coordinates.iter().all(|x| x.is_finite()));
        if location >= self.coordinates.len() {
            self.coordinates.resize(location + 1, [f64::NAN; DIMS]);
        }
        self.coordinates[location] = *coordinates;
        self._touched.push(location);
    }

    pub fn touched(&self) -> &[usize] {
        &self._touched
    }

    pub fn update(&mut self) {
        while let Some(location) = self._touched.pop() {
            // Special case for the root.
            if self.tree.len() == 0 {
                self.tree.push(Node::Leaf(location));
                continue;
            }
            // Recursive descent through the KD Tree to find insertion point.
            let mut cursor = Node::root();
            let tree_depth = Node::depth(self.tree.len() - 1);
            loop {
                if Node::depth(cursor) + 1 > tree_depth {
                    // If inserting the new node would exceed the current memory
                    // allocation, then rebuild the whole table to be more balanced.
                    self.rebuild_tree();
                    self._touched.clear();
                    break;
                }
                match self.tree[cursor] {
                    Node::Partition(split_plane) => {
                        let axis = Node::depth(cursor) % DIMS;
                        let [lesser, greater] = Node::children(cursor);
                        // Determine which side of the splitting plane the query
                        // point is on.
                        let (closer, _further) = if self.coordinates[location][axis] < split_plane {
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
                        self.build_tree_recursive(cursor, &mut [leaf, location]);
                        break;
                    }
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
            self.coordinates.len().next_power_of_two() * (1 + 1 + 2 + 4) - 1,
            Node::Leaf(usize::MAX), // Poison value.
        );
        let mut leaves: Vec<_> = (0..self.coordinates.len()).collect();
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
                self.coordinates[*x][axis]
                    .partial_cmp(&self.coordinates[*y][axis])
                    .unwrap()
            });
            // Find the largest element which is less than the pivot. Make the
            // splitting plane halfway between these two elements.
            let max_lesser = lesser.iter().fold(-f64::INFINITY, |max, idx| {
                max.max(self.coordinates[*idx][axis])
            });
            self.tree[node] = Node::Partition((self.coordinates[*pivot][axis] + max_lesser) / 2.0);
            // Recurse.
            let [child1, child2] = Node::children(node);
            let (lesser, greater) = leaves.split_at_mut(half);
            self.build_tree_recursive(child1, lesser);
            self.build_tree_recursive(child2, greater);
        }
    }

    pub fn search<'a>(&'a self, coordinates: &'a [f64; DIMS]) -> SearchIterator<'a, DIMS> {
        debug_assert!(self.touched().is_empty());
        let mut stack;
        if self.coordinates.is_empty() {
            stack = vec![]
        } else {
            stack = Vec::with_capacity(Node::depth(self.coordinates.len() - 1) - 1);
            stack.push(StackData {
                node: Node::root(),
                corner: *coordinates,
            });
        };
        return SearchIterator {
            kd_tree: self,
            coordinates,
            stack,
        };
    }

    pub fn nearest_neighbors(
        &self,
        k: usize,
        coordinates: &[f64; DIMS],
        distance_function: &impl Fn(&[f64; DIMS], &[f64; DIMS]) -> f64,
        mut max_distance: f64,
    ) -> Vec<(usize, f64)> {
        debug_assert!(k > 0);
        // Neighbors is priority queue, sorted by distance.
        let mut neighbors: Vec<(usize, f64)> = Vec::with_capacity(k);
        let mut search = self.search(coordinates);
        while let Some((location, dist)) = search.next(distance_function, max_distance) {
            // Insert this point into the nearest neighbors priority queue.
            if neighbors.len() == k {
                neighbors.pop();
            }
            let sorted_position = neighbors
                .iter()
                .position(|(_, nn_dist)| dist < *nn_dist)
                .unwrap_or(neighbors.len());
            neighbors.insert(sorted_position, (location, dist));
            max_distance = *neighbors
                .get(k - 1)
                .map(|(_location, distance)| distance)
                .unwrap_or(&max_distance);
        }
        return neighbors;
    }
}

#[derive(Debug, Copy, Clone)]
struct StackData<const DIMS: usize> {
    node: usize,
    corner: [f64; DIMS],
}

pub struct SearchIterator<'a, const DIMS: usize> {
    kd_tree: &'a KDTree<DIMS>,
    coordinates: &'a [f64; DIMS],
    stack: Vec<StackData<DIMS>>,
}

impl<'a, const DIMS: usize> SearchIterator<'a, DIMS> {
    pub fn next(
        &mut self,
        distance: &impl Fn(&[f64; DIMS], &[f64; DIMS]) -> f64,
        max_distance: f64,
    ) -> Option<(usize, f64)> {
        while let Some(mut cursor) = self.stack.pop() {
            // Check if this node can be discarded because its bounding box is
            // too far from the query point.
            if distance(&self.coordinates, &cursor.corner) >= max_distance {
                continue;
            }
            // Descend until it reaches a leaf.
            loop {
                match self.kd_tree.tree[cursor.node] {
                    Node::Partition(split_plane) => {
                        let axis = Node::depth(cursor.node) % DIMS;
                        let [lesser, greater] = Node::children(cursor.node);
                        // Determine which side of the splitting plane the query
                        // point is on.
                        let (closer, further) = if self.coordinates[axis] < split_plane {
                            (lesser, greater)
                        } else {
                            (greater, lesser)
                        };
                        // Push the further child onto the stack to deal with it later.
                        let mut new_corner = cursor.corner;
                        new_corner[axis] = split_plane;
                        self.stack.push(StackData {
                            node: further,
                            corner: new_corner,
                        });
                        // Move the cursor onto the closer child and continue descent.
                        cursor.node = closer;
                    }
                    Node::Leaf(index) => {
                        let dist = distance(&self.coordinates, &self.kd_tree.coordinates[index]);
                        if dist < max_distance {
                            return Some((index, dist));
                        } else {
                            break;
                        }
                    }
                }
            }
        }
        return None;
    }
}
