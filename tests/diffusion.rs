/*! Contrive a diffusion problem.

Solve with both a high resolution grid and sparser randomly placed nodes.

*/

const PIPE_LENGTH: f64 = 1000.0;
const PIPE_DIAMETER: f64 = 1.0;
const DIFFUSIVITY: f64 = 1.0;
const GRID_SPACING: f64 = 1.0;
const VORONOI_POINTS: usize = 10_000;
const TIME_STEP: f64 = 1e-3;
const ACCURACY: f64 = 1e-9;

use impulse_response::kd_tree::KDTree;
use impulse_response::sparse::{Model as SparseIRM, Vector as SparseVector};
use impulse_response::voronoi::ConvexHull;
use rayon::prelude::*;

struct Pipe {
    world: ConvexHull<3>,
    tree: KDTree<3>,
    voronoi_diagram: Vec<ConvexHull<3>>,
    irm: SparseIRM,
    emitters: Vec<usize>,
    concentration: Vec<f64>,
}

impl Pipe {
    fn new() -> Self {
        let mut pipe = Self {
            world: ConvexHull::aabb(
                &[0.0, 0.0, 0.0],
                &[PIPE_DIAMETER, PIPE_DIAMETER, PIPE_LENGTH],
            ),
            tree: Default::default(),
            voronoi_diagram: Default::default(),
            irm: SparseIRM::new(TIME_STEP, ACCURACY, f64::EPSILON),
            emitters: Default::default(),
            concentration: Default::default(),
        };
        for x in 0..(PIPE_DIAMETER / GRID_SPACING) as usize {
            for y in 0..(PIPE_DIAMETER / GRID_SPACING) as usize {
                let coords = [
                    (x as f64 + 0.5) * GRID_SPACING,
                    (y as f64 + 0.5) * GRID_SPACING,
                    0.0,
                ];
                let idx = pipe.tree.len();
                pipe.tree.touch(idx, &coords);
                pipe.irm.touch(idx);
                pipe.emitters.push(idx);
            }
        }
        return pipe;
    }

    fn grid_init(&mut self) {
        for z in 1..(PIPE_LENGTH / GRID_SPACING) as usize {
            for x in 0..(PIPE_DIAMETER / GRID_SPACING) as usize {
                for y in 0..(PIPE_DIAMETER / GRID_SPACING) as usize {
                    let coords = [
                        (x as f64 + 0.5) * GRID_SPACING,
                        (y as f64 + 0.5) * GRID_SPACING,
                        (z as f64 + 0.5) * GRID_SPACING,
                    ];
                    let idx = self.tree.len();
                    self.tree.touch(idx, &coords);
                    self.irm.touch(idx);
                }
            }
        }
        self.tree.update();
        self.voronoi_diagram = (0..self.tree.len())
            .into_par_iter()
            .map(|idx| ConvexHull::new(&self.tree, idx, &self.world))
            .collect();
        self.concentration.resize(self.tree.len(), 0.0);
    }

    fn voronoi_init(&mut self) {
        let emitters_reserve = GRID_SPACING * 1.5;
        for _ in 0..VORONOI_POINTS {
            let coords = [
                rand::random::<f64>() * PIPE_DIAMETER,
                rand::random::<f64>() * PIPE_DIAMETER,
                emitters_reserve + rand::random::<f64>() * (PIPE_LENGTH - emitters_reserve),
            ];
            let idx = self.tree.len();
            self.tree.touch(idx, &coords);
            self.irm.touch(idx);
        }
        self.tree.update();
        self.voronoi_diagram = (0..self.tree.len())
            .into_par_iter()
            .map(|idx| ConvexHull::new(&self.tree, idx, &self.world))
            .collect();
        self.concentration.resize(self.tree.len(), 0.0);
    }

    /// Apply an impulse response like input to all of the emitters.
    fn apply_input(&mut self) {
        let emitter_volume: f64 = self
            .emitters
            .iter()
            .map(|idx| self.voronoi_diagram[*idx].volume)
            .sum();
        let release_concentration = 1.0 / emitter_volume;
        dbg!(release_concentration);
        for idx in self.emitters.iter() {
            self.concentration[*idx] = release_concentration;
        }
    }

    fn advance(&mut self) {
        let mut next_concentration = vec![0.0; self.tree.len()];
        let Self {
            voronoi_diagram,
            irm,
            concentration,
            ..
        } = self;
        irm.advance(
            &concentration,
            &mut next_concentration,
            |state: &SparseVector, deriv: &mut SparseVector| {
                for idx in state.nonzero.iter().copied() {
                    let cell = &voronoi_diagram[idx];
                    let mut net_flux = 0.0;
                    for face in &cell.faces {
                        if let Some(adj) = face.facing_location {
                            let gradient = state.data[adj] - state.data[idx];
                            let flux = -DIFFUSIVITY * gradient * face.surface_area;
                            if state.data[adj] != 0.0 {
                                net_flux += flux;
                            } else {
                                if flux.abs() >= f64::EPSILON {
                                    net_flux += flux;
                                    let adj_volume = voronoi_diagram[adj].volume;
                                    deriv.data[adj] = -flux / adj_volume;
                                    deriv.nonzero.push(adj);
                                }
                            }
                        }
                    }
                    deriv.data[idx] = -net_flux / cell.volume;
                    deriv.nonzero.push(idx);
                }
            },
        );
        std::mem::swap(concentration, &mut next_concentration);
    }
}

// #[test]
fn diffusion() {
    let mut grid_model = Pipe::new();
    grid_model.grid_init();
    dbg!(grid_model.tree.len());

    let mut voronoi_model = Pipe::new();
    voronoi_model.voronoi_init();
    dbg!(voronoi_model.tree.len());

    grid_model.apply_input();
    voronoi_model.apply_input();

    for time_step in 0..=10_000 {
        grid_model.advance();
        voronoi_model.advance();
        if time_step > 100 && time_step % 100 != 0 {
            continue;
        }
        // Check that the models are approximately the same.
        let Pipe {
            tree,
            concentration,
            ..
        } = &mut grid_model;
        let interp =
            impulse_response::knn_interp::KnnInterpolator::<3, 1>::assemble(tree, unsafe {
                std::mem::transmute::<&mut Vec<f64>, &mut Vec<[f64; 1]>>(concentration)
            });
        let mut errors = vec![];
        for idx in voronoi_model.emitters.len()..voronoi_model.tree.len() {
            let coords = &voronoi_model.tree.coordinates[idx];
            let voronoi_results = voronoi_model.concentration[idx];
            let grid_results = interp.interpolate(coords, |_| panic!(), |_, _| panic!());
            errors.push((voronoi_results - grid_results[0]).abs());
        }
        let max_error = errors.iter().fold(-f64::INFINITY, |a, b| a.max(*b));
        dbg!(max_error);
        interp.disassemble(tree, unsafe { std::mem::transmute(concentration) });
    }
}
