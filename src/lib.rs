/*! # Impulse Response Integration Method

A tool for simulating dynamical systems.

This method works by measuring the impulse response of the system. If the system
is linear and time-invariant then the impulse response completely describes the
system. The exact state of the system can be computed by convolving the initial
state of the system with the impulse response. This method uses these facts to
solve integration problems with efficient matrix-vector multiplications.

### Details

Users specify their system with both:

* A state vector, which completely describes the system at an instant in time.

* The derivative of the state, as a function of the current state.

The users system is assumed to be linear and time-invariant.

This method uses the impulse response to advance the state of the system in
fixed time steps of length `time_step`. First compute the impulse response in
high fidelity using the Crank-Nicholson method with a variable length time-step.
Sample the response at `time_step` after the impulse. Measure the impulse
response of every state and store them in a matrix. Then to advance the state of
the integration, multiply that matrix by the state vector.

The impulse response matrix is a square matrix, and so its size is the length of
the state vector squared. Naively, this could cause performance issues for
systems which have a very large state vector. However in most systems with very
large state vectors: most of the states do not interact with each other over the
relatively short `time_step` at which it measures the impulse response. As a
result, the impulse responses are mostly zeros and the impulse response matrix
can be compressed into a sparse matrix.

If the users system has external inputs then extra steps are necessary to model
their interactions. Inputs are not represented in the state vector and the
impulse response function is not defined for inputs. Instead the impulse
response matrix is measured as a function of the inputs, which are assumed to be
constant over the time course of each integration time step. To efficiently
compute the function `ImpulseResponseMatrix(Inputs)`: first evaluate this
function at a great many input points; then construct an approximation from the
exactly known points. An advantage of approximate solutions over closed form
equations is that the approximation can handle non-linearities, which allows
this method to work for linear systems where the coefficients are non-linear
functions of the inputs.

The impulse response integration method runs fast, but can consume a significant
amount of time and memory at start up to compute and store the impulse
responses.

## Example: The Hodgkin Huxley Model

The Hodgkin Huxley model describes the initiation and propagation of action
potentials in neurons.

![](https://raw.githubusercontent.com/ctrl-z-9000-times/impulse_response/master/figures/hh_single_ap.png)
![](https://raw.githubusercontent.com/ctrl-z-9000-times/impulse_response/master/figures/hh_test_pattern.png)

Citation: Ryan Siciliano, 2012, The Hodgkin-Huxley Model: Its Extensions, Analysis and Numerics.

## Example: Measuring equivalent resistance

This comic strip poses an interesting problem. The problem does have a known
analytic solution, `4/pi - 1/2`, but it can also be approximated using numerical
methods. I demonstrate how to do this using the impulse response library.

[![](https://imgs.xkcd.com/comics/nerd_sniping.png)](https://xkcd.com/356/)

### Numerical Solution

* First alter the size of the grid of resistors, from an infinite grid to a very
large grid. Otherwise it would not be possible to compute! Because of this
change the resulting approximation will overestimate the true value. In the
limit, as the size of the grid approaches infinity, this overestimation error
approaches zero.

* Attach a capacitor to every node in the grid. This simulates the parasitic
capacitance which exists in all wires. Capacitance needs to be included in the
model in order to simulate the flow of electric charge.

* Connect a voltage source across the two marked nodes, and measure how much
current passes through the voltage source. Then compute the equivalent
resistance using the formula: `V = I R`. Since the system contains capacitors,
it will take time for the current to reach a steady state. Measuring the steady
state current entails simulating the system for a considerable amount of time.

### Implementation and Results

The source code is an annotated example of how to use this library.
Link: [impulse_response/examples/nerd_sniping.rs](https://github.com/ctrl-z-9000-times/impulse_response/blob/master/examples/nerd_sniping.rs)

Result of running the code with a 32x32 grid:
```text
$ time cargo run --example nerd_sniping --release
Model Size: 1024 Nodes
Equivalent Resistance: 0.8825786612296072 Ohms
Exact Answer: 4/PI - 1/2 = 0.7732395447351628 Ohms
```

Now lets increase the size of the grid to 633x633, and observe that the
equivalent resistance is closer to the correct value:
```text
$ cargo run --example nerd_sniping --release
Model Size: 400689 Nodes
Equivalent Resistance: 0.8416329950362197 Ohms
Exact Answer: 4/PI - 1/2 = 0.7732395447351628 Ohms
```
Runtime: 2 days.

The measurement error is approximately 8%.

## More Examples

* `tests/exponential_decay.rs`
    + Smoke test, simulates a system which simply decays exponentially.

* `examples/benchmark.rs`
    + An artificial scenario.
    + Demonstrates modifying a sparse system while its running.

* `tests/interpolation.rs`
    + Demonstrates the interpolation algorithm, which is a weighted average of a
    K-Nearest-Neighbors search using a KD-Tree.

* `tests/leaky_cable.rs`
    + Simulates the electricity in a neurons dendrite.

*/
#![feature(trait_alias)]
#![feature(const_generics)]
#![feature(const_int_pow)]
#![feature(is_sorted)]
#![feature(slice_partition_at_index)]
#![allow(incomplete_features)]
pub mod dense;
pub mod knn;
pub mod sparse;

enum IntegrationMethod {
    #[allow(dead_code)]
    ForwardEuler,

    #[allow(dead_code)]
    BackwardEuler {
        iterations: usize,
    },

    CrankNicholson {
        iterations: usize,
    },
}

enum IntegrationTimestep {
    #[allow(dead_code)]
    Constant(f64),

    Variable {
        error_tolerance: f64,
    },
}
