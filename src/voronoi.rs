/*! Voronoi Diagrams & Delaunay Triangulations */
use crate::kd_tree::KDTree;
use std::collections::HashMap;
use std::sync::Arc;

// TODO: Consider splitting this into two structures for the computation state &
// the outputs. Then I can switch the "Arc" to "Rc" because the user can never
// send them between threads.

// OR: remove the RC altogether by inlining the planes into the triangles. Then
// sort the triangles by facing_location so that it can skip the duplicate
// planes.

#[derive(Debug, Clone)]
pub struct ConvexHull<const DIMS: usize> {
    vertexes: Vec<[f64; DIMS]>,
    vertex_alive: Vec<bool>,
    /// Track of how far each vertex is from the home coordinates, to reduce the
    /// search space to the smallest neighborhood.
    vertex_distances: Vec<f64>,
    triangles: Vec<Triangle<DIMS>>,
    planes: Vec<Arc<Plane<DIMS>>>,
    pub volume: f64,
    pub faces: Vec<Face<DIMS>>,
}

#[derive(Debug, Clone)]
pub struct Face<const DIMS: usize> {
    pub facing_location: Option<usize>,
    // pub outline: Vec<[f64; DIMS]>,
    pub surface_area: f64,
    // plane: Arc<Plane<DIMS>>,
}

#[derive(Debug, Clone)]
struct Triangle<const DIMS: usize> {
    vertexes: [usize; 3],
    facing_location: usize,
    plane: Arc<Plane<DIMS>>,
}

impl<const DIMS: usize> ConvexHull<DIMS> {
    pub fn new(
        kd_tree: &KDTree<DIMS>,
        location: usize,
        world_boundaries: &ConvexHull<DIMS>,
    ) -> Self {
        let home_coordinates = &kd_tree.coordinates[location];
        debug_assert!(world_boundaries.contains_point(home_coordinates));
        let mut hull = world_boundaries.clone();
        hull.vertex_alive = vec![true; hull.vertexes.len()];
        hull.faces.clear();
        let mut nn_search = kd_tree.search(&home_coordinates);
        while let Some((neighbor, _distance)) = nn_search.next(
            &distance_function,
            2.0 * hull.furthest_extent(&home_coordinates),
        ) {
            if neighbor == location {
                continue;
            }
            // Make a dividing plane halfway between these locations.
            hull.add_plane(
                &Plane::bisect_points(home_coordinates, &kd_tree.coordinates[neighbor]),
                neighbor,
            );
        }
        hull.finalize(home_coordinates);
        // Free all of the private working data.
        std::mem::take(&mut hull.vertexes);
        std::mem::take(&mut hull.vertex_alive);
        std::mem::take(&mut hull.vertex_distances);
        std::mem::take(&mut hull.triangles);
        return hull;
    }

    pub fn contains_point(&self, coordinates: &[f64; DIMS]) -> bool {
        return self.planes.iter().all(|p| p.contains_point(coordinates));
    }

    fn furthest_extent(&mut self, home: &[f64; DIMS]) -> f64 {
        for (v, dist) in self.vertex_distances.iter_mut().enumerate() {
            if dist.is_nan() {
                *dist = distance_function(home, &self.vertexes[v]);
            }
        }
        while self.vertex_distances.len() < self.vertexes.len() {
            self.vertex_distances.push(distance_function(
                home,
                &self.vertexes[self.vertex_distances.len()],
            ));
        }
        self.vertex_distances
            .iter()
            .enumerate()
            .filter_map(|(idx, distance)| {
                if self.vertex_alive[idx] {
                    Some(distance)
                } else {
                    None
                }
            })
            .fold(-f64::INFINITY, |accum, distance| accum.max(*distance))
    }

    /// Removes all areas of the convex hull which are above the given plane. If
    /// the entire convex hull is below the plane, then this does nothing. Use
    /// this method to sculpt an axis aligned bounding box into any desired
    /// convex shape.
    fn add_plane(&mut self, plane: &Plane<DIMS>, facing_location: usize) {
        let remove_indexes: Vec<_> = (0..self.vertexes.len())
            .filter(|&v_idx| self.vertex_alive[v_idx])
            .filter(|&v_idx| !plane.contains_point(&self.vertexes[v_idx]))
            .collect();
        if remove_indexes.is_empty() {
            return;
        }
        let mut freelist: Vec<_> = self
            .vertex_alive
            .iter()
            .enumerate()
            .filter_map(|(idx, alive)| if *alive { None } else { Some(idx) })
            .collect();
        for v in &remove_indexes {
            self.vertex_alive[*v] = false;
        }
        let plane = Arc::new(plane.clone());
        self.planes.push(plane.clone());
        let Self {
            vertexes,
            vertex_alive,
            triangles,
            ..
        } = self;
        let mut new_vertex_pairs: Vec<(usize, usize)> = vec![];
        // Cache all line-plane intersection points, so that duplicate
        // calculations yield a consolidated result.
        let mut intersection_cache = HashMap::<(usize, usize), usize>::new();
        let mut intersection = |keep: usize, remove: usize| {
            if let Some(x) = intersection_cache.get(&(keep, remove)) {
                return *x;
            }
            let x = plane.line_intersection(&vertexes[keep], &vertexes[remove]);
            // Calculate the line-plane intersection.
            let x_index = match freelist.pop() {
                Some(x_index) => {
                    vertexes[x_index] = x;
                    x_index
                }
                None => {
                    let x_index = vertexes.len();
                    vertexes.push(x);
                    x_index
                }
            };
            intersection_cache.insert((keep, remove), x_index);
            return x_index;
        };
        // Determine how the plane intersects each triangle, and update the
        // triangles such that they are all contained within the given plane.
        for triangle_idx in (0..triangles.len()).rev() {
            let triangle = &mut triangles[triangle_idx];
            let three_v = triangle.vertexes;
            debug_assert!(three_v[0] != three_v[1]);
            debug_assert!(three_v[0] != three_v[2]);
            debug_assert!(three_v[1] != three_v[2]);
            let num_alive = three_v.iter().filter(|&v| vertex_alive[*v]).count();
            match num_alive {
                3 => {
                    // Triangle is untouched by the plane, do nothing.
                }
                2 => {
                    // One corner removed, triangle now has 4 sides. Break into
                    // two triangles.
                    let remove_vertex = *three_v
                        .iter()
                        .filter(|&v| !vertex_alive[*v])
                        .next()
                        .unwrap();
                    let mut iter = three_v.iter().filter(|&v| vertex_alive[*v]);
                    let keep_vertex_1 = *iter.next().unwrap();
                    let keep_vertex_2 = *iter.next().unwrap();
                    let new_vertex_1 = intersection(keep_vertex_1, remove_vertex);
                    let new_vertex_2 = intersection(keep_vertex_2, remove_vertex);
                    new_vertex_pairs.push((new_vertex_1, new_vertex_2));
                    // Update the triangle, and make a new triangle too.
                    triangle.vertexes = [keep_vertex_1, keep_vertex_2, new_vertex_2];
                    let mut new_triangle = triangle.clone();
                    new_triangle.vertexes = [keep_vertex_1, new_vertex_1, new_vertex_2];
                    triangles.push(new_triangle);
                }
                1 => {
                    // Two corners removed, shrink the triangle.
                    let keep_vertex = *three_v.iter().filter(|&v| vertex_alive[*v]).next().unwrap();
                    let mut iter = three_v.iter().filter(|&v| !vertex_alive[*v]);
                    let remove_vertex_1 = *iter.next().unwrap();
                    let remove_vertex_2 = *iter.next().unwrap();
                    let new_vertex_1 = intersection(keep_vertex, remove_vertex_1);
                    let new_vertex_2 = intersection(keep_vertex, remove_vertex_2);
                    triangle.vertexes = [keep_vertex, new_vertex_1, new_vertex_2];
                    new_vertex_pairs.push((new_vertex_1, new_vertex_2));
                }
                0 => {
                    // Triangle is entirely above the plane, remove it.
                    triangles.swap_remove(triangle_idx);
                }
                _ => panic!(),
            }
        }
        vertex_alive.resize(self.vertexes.len(), true);
        for (a, b) in &new_vertex_pairs {
            vertex_alive[*a] = true;
            vertex_alive[*b] = true;
        }
        self.vertex_distances.resize(self.vertexes.len(), f64::NAN);
        for (a, b) in &new_vertex_pairs {
            self.vertex_distances[*a] = f64::NAN;
            self.vertex_distances[*b] = f64::NAN;
        }
        // Sort the new vertexes into a polygon ring. Actually, allow multiple
        // polygons. In theory, the intersection between a plane and a convex
        // hull should be at most one convex polygon, but due to floating point
        // issues the hull is not always strictly convex.
        while !new_vertex_pairs.is_empty() {
            let mut outline = Vec::with_capacity(new_vertex_pairs.len());
            let (_a, b) = new_vertex_pairs.pop().unwrap();
            outline.push(b);
            for _ in 0..new_vertex_pairs.len() {
                for idx in 0..new_vertex_pairs.len() {
                    let (a, b) = new_vertex_pairs[idx];
                    if a == *outline.last().unwrap() {
                        outline.push(b);
                        new_vertex_pairs.swap_remove(idx);
                        break;
                    } else if b == *outline.last().unwrap() {
                        outline.push(a);
                        new_vertex_pairs.swap_remove(idx);
                        break;
                    }
                }
            }
            // Triangulate the new vertexes into a surface of triangles.
            let v1 = outline[0];
            for idx in 1..outline.len() - 1 {
                let v2 = outline[idx];
                let v3 = outline[idx + 1];
                self.triangles.push(Triangle {
                    vertexes: [v1, v2, v3],
                    facing_location,
                    plane: plane.clone(),
                });
            }
        }
        // Cull unused planes.
        for p_idx in (0..self.planes.len()).rev() {
            if Arc::strong_count(&self.planes[p_idx]) == 1 {
                self.planes.swap_remove(p_idx);
            }
        }
    }

    fn finalize(&mut self, interior_coordinates: &[f64; DIMS]) {
        // Consolidate the Triangles into Faces.
        self.volume = 0.0;
        for t in &self.triangles {
            self.volume += t.volume(&self.vertexes, interior_coordinates);
        }
        self.triangles.sort_by_key(|t| t.facing_location);
        let mut prev = None;
        for t in &self.triangles {
            if Some(t.facing_location) == prev {
                let f = self.faces.last_mut().unwrap();
                f.surface_area += t.surface_area(&self.vertexes);
            // f.outline.extend_from_slice(t.vertexes);
            } else {
                prev = Some(t.facing_location);
                self.faces.push(Face {
                    facing_location: if t.facing_location >= usize::MAX - 6 {
                        None // Facing the edge of the world.
                    } else {
                        Some(t.facing_location)
                    },
                    // outline: vec![],
                    // plane: t.plane.clone(),
                    surface_area: t.surface_area(&self.vertexes),
                });
            }
        }
    }
}

impl ConvexHull<3> {
    /// Axis Aligned Bounding Box
    pub fn aabb(lower_corner: &[f64; 3], upper_corner: &[f64; 3]) -> Self {
        const DIMS: usize = 3;
        let mut hull = Self {
            vertexes: vec![],
            vertex_alive: vec![],
            vertex_distances: vec![],
            planes: vec![],
            triangles: vec![],
            faces: vec![],
            volume: f64::NAN,
        };
        let num_corners: usize = 2_usize.pow(DIMS as u32);
        for x in 0..num_corners {
            let mut coordinates = *lower_corner;
            for d in 0..DIMS {
                if x & (1 << d) != 0 {
                    coordinates[d] = upper_corner[d];
                }
            }
            hull.vertexes.push(coordinates);
        }
        for d in 0..DIMS {
            let mut normal = [0.0; DIMS];
            normal[d] = -1.0;
            let plane = Arc::new(Plane::point_normal(lower_corner, &normal));
            hull.planes.push(plane.clone());
            for _ in 0..2 {
                hull.triangles.push(Triangle {
                    plane: plane.clone(),
                    vertexes: [usize::MAX; 3],
                    facing_location: usize::MAX - (2 * d),
                });
            }
            normal[d] = 1.0;
            let plane = Arc::new(Plane::point_normal(upper_corner, &normal));
            hull.planes.push(plane.clone());
            for _ in 0..2 {
                hull.triangles.push(Triangle {
                    plane: plane.clone(),
                    vertexes: [usize::MAX; 3],
                    facing_location: usize::MAX - (2 * d + 1),
                });
            }
        }
        // DIM 0 -
        hull.triangles[0].vertexes[0] = 0;
        hull.triangles[0].vertexes[1] = 6;
        hull.triangles[0].vertexes[2] = 2;
        hull.triangles[1].vertexes[0] = 0;
        hull.triangles[1].vertexes[1] = 6;
        hull.triangles[1].vertexes[2] = 4;
        // DIM 0 +
        hull.triangles[2].vertexes[0] = 1;
        hull.triangles[2].vertexes[1] = 7;
        hull.triangles[2].vertexes[2] = 3;
        hull.triangles[3].vertexes[0] = 1;
        hull.triangles[3].vertexes[1] = 7;
        hull.triangles[3].vertexes[2] = 5;
        // DIMS 1 -
        hull.triangles[4].vertexes[0] = 0;
        hull.triangles[4].vertexes[1] = 1;
        hull.triangles[4].vertexes[2] = 4;
        hull.triangles[5].vertexes[0] = 1;
        hull.triangles[5].vertexes[1] = 4;
        hull.triangles[5].vertexes[2] = 5;
        // DIMS 1 +
        hull.triangles[6].vertexes[0] = 2;
        hull.triangles[6].vertexes[1] = 3;
        hull.triangles[6].vertexes[2] = 6;
        hull.triangles[7].vertexes[0] = 3;
        hull.triangles[7].vertexes[1] = 6;
        hull.triangles[7].vertexes[2] = 7;
        // DIMS 2 -
        hull.triangles[8].vertexes[0] = 0;
        hull.triangles[8].vertexes[1] = 2;
        hull.triangles[8].vertexes[2] = 3;
        hull.triangles[9].vertexes[0] = 3;
        hull.triangles[9].vertexes[1] = 1;
        hull.triangles[9].vertexes[2] = 0;
        // DIMS 2 +
        hull.triangles[10].vertexes[0] = 7;
        hull.triangles[10].vertexes[1] = 4;
        hull.triangles[10].vertexes[2] = 6;
        hull.triangles[11].vertexes[0] = 4;
        hull.triangles[11].vertexes[1] = 7;
        hull.triangles[11].vertexes[2] = 5;
        // Make an arbitrary interior point for volume calculation.
        let mut home = [0.0; DIMS];
        for v in &hull.vertexes {
            for d in 0..DIMS {
                home[d] += v[d];
            }
        }
        for d in 0..DIMS {
            home[d] /= hull.vertexes.len() as f64;
        }
        debug_assert!(hull.contains_point(&home));
        hull.finalize(&home);
        return hull;
    }
    // TODO: Methods for making and manipulating convex hulls, for
    // positioning the worlds bounding box over the users data.
    //      * shifting, scaling, and rotating a hull.
    //      * common geometric shapes: sphere, cylinder.
}

impl<const DIMS: usize> Triangle<DIMS> {
    /// Compute the volume of the triangular pyramid formed by this triangle and
    /// the given point (which should be inside of the convex hull).
    fn volume(&self, coordinates: &[[f64; DIMS]], home: &[f64; DIMS]) -> f64 {
        // Put the anchor at the origin.
        let [v_anchor, v_b, v_c] = self.vertexes;
        let vertex_a = coordinates[v_anchor];
        let vertex_b = coordinates[v_b];
        let vertex_c = coordinates[v_c];
        let vector_b = subtract(&vertex_b, &vertex_a);
        let vector_c = subtract(&vertex_c, &vertex_a);
        let vector_home = subtract(&home, &vertex_a);
        let base_cross = cross_product(&vector_b, &vector_c);
        let base_cross_magnitude = magnitude(&base_cross);
        if base_cross_magnitude == 0.0 {
            return 0.0;
        }
        let mut base_normal = base_cross;
        for d in 0..DIMS {
            base_normal[d] /= base_cross_magnitude;
        }
        let height = dot_product(&base_normal, &vector_home);
        let base_area = 0.5 * base_cross_magnitude;
        return f64::abs(height * base_area / 3.0);
    }

    fn surface_area(&self, coordinates: &[[f64; DIMS]]) -> f64 {
        let vertex_a = &coordinates[self.vertexes[0]];
        let vertex_b = &coordinates[self.vertexes[1]];
        let vertex_c = &coordinates[self.vertexes[2]];
        let vector_ab = subtract(vertex_b, vertex_a);
        let vector_ac = subtract(vertex_c, vertex_a);
        return 0.5 * magnitude(&cross_product(&vector_ab, &vector_ac));
    }
}

/// Strictly Euclidean geometry.
fn distance_function<const DIMS: usize>(a: &[f64; DIMS], b: &[f64; DIMS]) -> f64 {
    f64::sqrt(
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| f64::powi(x - y, 2))
            .sum(),
    )
}

fn subtract<const DIMS: usize>(a: &[f64; DIMS], b: &[f64; DIMS]) -> [f64; DIMS] {
    let mut diff = [0.0; DIMS];
    for d in 0..DIMS {
        diff[d] = a[d] - b[d];
    }
    return diff;
}

fn magnitude<const DIMS: usize>(vector: &[f64; DIMS]) -> f64 {
    distance_function(vector, &[0.0; DIMS])
}

fn dot_product<const DIMS: usize>(a: &[f64; DIMS], b: &[f64; DIMS]) -> f64 {
    return a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
}

fn cross_product<const DIMS: usize>(a: &[f64; DIMS], b: &[f64; DIMS]) -> [f64; DIMS] {
    let mut x = [0.0; DIMS];
    match DIMS {
        3 => {
            x[0] = a[1] * b[2] - a[2] * b[1];
            x[1] = a[2] * b[0] - a[0] * b[2];
            x[2] = a[0] * b[1] - a[1] * b[0];
        }
        _ => unimplemented!(),
    }
    return x;
}

#[derive(Debug, Copy, Clone)]
struct Plane<const DIMS: usize> {
    normal: [f64; DIMS],
    offset: f64,
}

impl<const DIMS: usize> Plane<DIMS> {
    fn point_normal(point: &[f64; DIMS], normal: &[f64; DIMS]) -> Self {
        let mut normal = normal.clone();
        let normalize = magnitude(&normal);
        debug_assert!(normalize > 0.0);
        for x in &mut normal {
            *x /= normalize;
        }
        return Plane {
            normal,
            offset: dot_product(point, &normal),
        };
    }

    fn bisect_points(under: &[f64; DIMS], over: &[f64; DIMS]) -> Self {
        let mut midpoint = [0.0; DIMS];
        for d in 0..DIMS {
            midpoint[d] = 0.5 * (under[d] + over[d]);
        }
        return Plane::point_normal(&midpoint, &subtract(over, under));
    }

    /// Is the point contained within or below the plane?
    fn contains_point(&self, point: &[f64; DIMS]) -> bool {
        let tolerance = f64::EPSILON * (DIMS * DIMS * 1000) as f64;
        return dot_product(&self.normal, point) <= self.offset + tolerance;
    }

    fn line_intersection(&self, a: &[f64; DIMS], b: &[f64; DIMS]) -> [f64; DIMS] {
        let l_point = a;
        let l_direction = subtract(b, a);
        let denom = dot_product(&self.normal, &l_direction);
        let numer = self.offset - dot_product(&self.normal, l_point);
        assert!(denom != 0.0);
        let x = (numer / denom).max(0.0).min(1.0);
        let mut intersection = l_direction;
        for d in 0..DIMS {
            intersection[d] *= x;
            intersection[d] += l_point[d];
        }
        return intersection;
    }
}
