// TODO: Find the original license for this file ...

use std::iter::Enumerate;
use std::iter::Zip;

pub struct NN {
    num_inputs: u32,
    layers: Vec<Vec<Vec<f64>>>,
    learning_rate: f64,
    momentum: f64,
    momentum_vector: Vec<Vec<Vec<f64>>>,
}

impl NN {
    /// Each number in the `layers_sizes` parameter specifies a layer in the
    /// network. The number itself is the number of nodes in that layer. The
    /// first number is the input layer, the last number is the output layer,
    /// and all numbers between the first and last are hidden layers. There must
    /// be at least two layers in the network.
    pub fn new(layers_sizes: &[u32], learning_rate: f64, momentum: f64) -> NN {
        assert!(layers_sizes.len() >= 2, "Must have at least two layers!");
        assert!(
            layers_sizes.iter().all(|&x| x > 0),
            "Can't have any empty layers!"
        );
        assert!(
            learning_rate > 0f64,
            "The learning rate must be a positive number!"
        );
        assert!(momentum >= 0f64, "The momentum must be a positive number!");
        NN {
            num_inputs: layers_sizes[0],
            layers: NN::init_vector(layers_sizes, || rand::random::<f64>() - 0.5),
            learning_rate,
            momentum,
            momentum_vector: NN::init_vector(layers_sizes, || 0f64),
        }
    }

    fn init_vector<F: Fn() -> f64>(layers_sizes: &[u32], fill_value: F) -> Vec<Vec<Vec<f64>>> {
        let mut layers = Vec::with_capacity(layers_sizes.len() - 1);
        for (&prev_layer_size, &layer_size) in layers_sizes.iter().zip(layers_sizes.iter().skip(1))
        {
            let mut layer: Vec<Vec<f64>> = Vec::with_capacity(layer_size as usize);
            for _ in 0..layer_size {
                let mut node: Vec<f64> = Vec::with_capacity(prev_layer_size as usize + 1);
                for _ in 0..prev_layer_size + 1 {
                    node.push(fill_value());
                }
                layer.push(node)
            }
            layers.push(layer);
        }
        layers
    }

    /// Runs the network on an input and returns all activity organized in a
    /// vector of vectors. The outer vector is for layers. Get of output layer
    /// activity with `nn.run(input).pop().unwrap()`.
    pub fn run(&self, inputs: &[f64]) -> Vec<Vec<f64>> {
        debug_assert!(
            inputs.len() as u32 == self.num_inputs,
            "Input has a different length than the network's input layer!"
        );
        let mut results = Vec::new();
        results.push(inputs.to_vec());
        for (layer_index, layer) in self.layers.iter().enumerate() {
            let mut layer_results = Vec::new();
            for node in layer.iter() {
                layer_results.push(sigmoid(modified_dotprod(&node, &results[layer_index])))
            }
            results.push(layer_results);
        }
        results
    }

    pub fn train(&mut self, results: &Vec<Vec<f64>>, targets: &[f64], sample_weight: f64) -> f64 {
        debug_assert!(
            targets.len() == self.layers[self.layers.len() - 1].len(),
            "Target output has a different length than the network's output layer!"
        );
        let weight_updates = self.calculate_weight_updates(results, targets);
        let learning_rate = self.learning_rate * sample_weight;
        // Updates all weights in the network
        for layer_index in 0..self.layers.len() {
            let layer = &mut self.layers[layer_index];
            let layer_weight_updates = &weight_updates[layer_index];
            for node_index in 0..layer.len() {
                let node = &mut layer[node_index];
                let node_weight_updates = &layer_weight_updates[node_index];
                for weight_index in 0..node.len() {
                    let weight_update = node_weight_updates[weight_index];
                    let prev_delta = self.momentum_vector[layer_index][node_index][weight_index];
                    let delta = (learning_rate * weight_update) + (self.momentum * prev_delta);
                    node[weight_index] += delta;
                    self.momentum_vector[layer_index][node_index][weight_index] = delta;
                }
            }
        }
        -1.
    }

    fn calculate_weight_updates(
        &self,
        results: &Vec<Vec<f64>>,
        targets: &[f64],
    ) -> Vec<Vec<Vec<f64>>> {
        let mut network_errors: Vec<Vec<f64>> = Vec::new();
        let mut network_weight_updates = Vec::new();
        let layers = &self.layers;
        let network_results = &results[1..]; // skip the input layer
        let mut next_layer_nodes: Option<&Vec<Vec<f64>>> = None;

        for (layer_index, (layer_nodes, layer_results)) in
            iter_zip_enum(layers, network_results).rev()
        {
            let prev_layer_results = &results[layer_index];
            let mut layer_errors = Vec::new();
            let mut layer_weight_updates = Vec::new();

            for (node_index, (node, &result)) in iter_zip_enum(layer_nodes, layer_results) {
                let mut node_weight_updates = Vec::new();

                // calculate error for this node
                let node_error = if layer_index == layers.len() - 1 {
                    result * (1f64 - result) * (targets[node_index] - result)
                } else {
                    let mut sum = 0f64;
                    let next_layer_errors = &network_errors[network_errors.len() - 1];
                    for (next_node, &next_node_error_data) in next_layer_nodes
                        .unwrap()
                        .iter()
                        .zip((next_layer_errors).iter())
                    {
                        // +1 because the 0th weight is the threshold
                        sum += next_node[node_index + 1] * next_node_error_data;
                    }
                    result * (1f64 - result) * sum
                };

                // calculate weight updates for this node
                for weight_index in 0..node.len() {
                    let prev_layer_result = if weight_index == 0 {
                        1f64 // threshold
                    } else {
                        prev_layer_results[weight_index - 1]
                    };
                    let weight_update = node_error * prev_layer_result;
                    node_weight_updates.push(weight_update);
                }

                layer_errors.push(node_error);
                layer_weight_updates.push(node_weight_updates);
            }

            network_errors.push(layer_errors);
            network_weight_updates.push(layer_weight_updates);
            next_layer_nodes = Some(&layer_nodes);
        }

        // Updates were built by backpropagation so reverse them.
        network_weight_updates.reverse();

        network_weight_updates
    }
}

fn modified_dotprod(node: &Vec<f64>, values: &Vec<f64>) -> f64 {
    let mut it = node.iter();
    let mut total = *it.next().unwrap(); // start with the threshold weight
    for (weight, value) in it.zip(values.iter()) {
        total += weight * value;
    }
    total
}

fn sigmoid(y: f64) -> f64 {
    1f64 / (1f64 + (-y).exp())
}

// takes two arrays and enumerates the iterator produced by zipping each of
// their iterators together
fn iter_zip_enum<'s, 't, S: 's, T: 't>(
    s: &'s [S],
    t: &'t [T],
) -> Enumerate<Zip<std::slice::Iter<'s, S>, std::slice::Iter<'t, T>>> {
    s.iter().zip(t.iter()).enumerate()
}
