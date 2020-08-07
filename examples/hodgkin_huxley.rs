/*! Hodgkin-Huxley Model

Run using:
`$ python examples/hodgkin_huxley.py`
*/
use std::io::BufRead;
use std::sync::Arc;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt()]
struct CLI {
    #[structopt(short, long = "time_step", default_value = "1e-3")]
    time_step: f64,

    #[structopt(long)]
    numeric: bool,

    #[structopt(long)]
    train: bool,
}

mod training_patterns {
    use std::f64::consts::TAU;
    pub trait Waveform: std::fmt::Debug {
        fn eval(&self, milliseconds: f64) -> f64;
    }

    #[derive(Debug)]
    pub struct SinWave {
        period: f64,
        amplitude: f64,
        phase: f64,
    }
    impl SinWave {
        pub fn new() -> SinWave {
            SinWave {
                period: rand::random::<f64>() * 2_000.0 + 500.0, // milliseconds.
                amplitude: rand::random::<f64>() * 0.03 + 0.002, // milliamps.
                phase: rand::random::<f64>() * TAU,
            }
        }
    }
    impl Waveform for SinWave {
        fn eval(&self, milliseconds: f64) -> f64 {
            f64::sin(TAU * milliseconds / self.period + self.phase) * self.amplitude
        }
    }

    #[derive(Debug)]
    pub struct SquareWave {
        period: f64,
        amplitude: f64,
        phase: f64,
        duty_cycle: f64,
    }
    impl SquareWave {
        pub fn new() -> SquareWave {
            SquareWave {
                period: rand::random::<f64>() * 2_000.0 + 500.0, // milliseconds.
                amplitude: rand::random::<f64>() * 0.03 + 0.002, // milliamps.
                phase: rand::random::<f64>(),
                duty_cycle: rand::random::<f64>() * 0.15,
            }
        }
    }
    impl Waveform for SquareWave {
        fn eval(&self, milliseconds: f64) -> f64 {
            let angle = (milliseconds / self.period + self.phase) % 1.0;
            if angle <= self.duty_cycle {
                self.amplitude
            } else {
                0.0
            }
        }
    }
}

// Model Parameters.
const C: f64 = 0.01; // uF / cm^2

// Leak Current.
const E_LEAK: f64 = -49.42; // mV
const G_LEAK: f64 = 0.003; // mS / cm^2

// Voltage Gated Sodium Channels.
type NavModel = impulse_response::dense::Model<1, 1, 4, 16, 5>;
type NavInstance = impulse_response::dense::Instance<1, 1, 4, 16, 5>;
const E_NA: f64 = 55.17; // mV
const G_NA: f64 = 1.2; // mS / cm^2
const M_INV: usize = 0; // Enumerate the states.
const M: usize = 1;
const H_INV: usize = 2;
const H: usize = 3;

// Voltage Gated Potassium Channels.
type KModel = impulse_response::dense::Model<1, 1, 2, 4, 3>;
type KInstance = impulse_response::dense::Instance<1, 1, 2, 4, 3>;
const E_K: f64 = -72.14; // mV
const G_K: f64 = 0.36; // mS / cm^2
const N_INV: usize = 0; // Enumerate the states.
const N: usize = 1;

struct State {
    voltage: f64,
    nav: NavInstance,
    kv: KInstance,
}

impl State {
    fn new(nav_class: Arc<NavModel>, kv_class: Arc<KModel>) -> State {
        State {
            voltage: -60.0, // mV
            nav: impulse_response::dense::Instance::new(&nav_class, &[1.0, 0.0, 1.0, 0.0]),
            kv: impulse_response::dense::Instance::new(&kv_class, &[1.0, 0.0]),
        }
    }

    fn advance_numeric(&mut self, i_inject: f64) {
        let i_na = self.nav.advance_numeric([self.voltage])[0] * (self.voltage - E_NA);
        let i_k = self.kv.advance_numeric([self.voltage])[0] * (self.voltage - E_K);
        let i_leak = G_LEAK * (self.voltage - E_LEAK);
        let i_net = i_inject - i_na - i_k - i_leak;
        self.voltage += i_net * self.kv.model.time_step / C;
    }

    fn advance(&mut self, i_inject: f64) {
        let i_na = self.nav.advance([self.voltage])[0] * (self.voltage - E_NA);
        let i_k = self.kv.advance([self.voltage])[0] * (self.voltage - E_K);
        let i_leak = G_LEAK * (self.voltage - E_LEAK);
        let i_net = i_inject - i_na - i_k - i_leak;
        self.voltage += i_net * self.kv.model.time_step / C;
    }
}

fn main() {
    let args = CLI::from_args();

    let state_drift = 1e-3;
    let voltage_tolerance = 1e-6;
    let conductance_tolerance = voltage_tolerance / 10.0;

    let nav_class = NavModel::new(
        args.time_step,
        Box::new(|inputs, state| {
            let v = inputs[0];
            let alpha_m = 0.1 * (v + 35.0) / (1.0 - (-(v + 35.0) / 10.0).exp());
            let beta_m = 4.0 * (-0.0556 * (v + 60.0)).exp();
            let alpha_h = 0.07 * (-0.05 * (v + 60.0)).exp();
            let beta_h = 1.0 / ((-0.1 * (v + 30.0)).exp() + 1.0);
            let deriv_m = alpha_m * state[M_INV] - beta_m * state[M];
            let deriv_h = alpha_h * state[H_INV] - beta_h * state[H];
            [-deriv_m, deriv_m, -deriv_h, deriv_h]
        }),
        Box::new(|state| [G_NA * state[M].powi(3) * state[H]]),
        Some(Box::new(|state| {
            let sum_m = state[M_INV] + state[M];
            let sum_h = state[H_INV] + state[H];
            if sum_m != 0.0 {
                state[M_INV] /= sum_m;
                state[M] /= sum_m;
            }
            if sum_h != 0.0 {
                state[H_INV] /= sum_h;
                state[H] /= sum_h;
            }
        })),
        &[voltage_tolerance; 1],
        &[conductance_tolerance; 1],
        &[state_drift; 4],
    );

    let kv_class = KModel::new(
        args.time_step,
        Box::new(|inputs, state| {
            let v = inputs[0];
            let alpha_n = 0.01 * (v + 50.0) / (1.0 - (-(v + 50.0) / 10.0).exp());
            let beta_n = 0.125 * (-(v + 60.0) / 80.0).exp();
            let deriv_n = alpha_n * state[N_INV] - beta_n * state[N];
            [-deriv_n, deriv_n]
        }),
        Box::new(|state| [G_K * state[N].powi(4)]),
        Some(Box::new(|state| {
            let sum: f64 = state.iter().sum();
            state.iter_mut().for_each(|x| *x /= sum);
        })),
        &[voltage_tolerance; 1],
        &[conductance_tolerance; 1],
        &[state_drift; 2],
    );

    // Train the model.
    if args.train {
        // Train for steady state.
        let mut state = State::new(nav_class.clone(), kv_class.clone());
        for _ in 0..(5_000.0 / args.time_step) as usize {
            state.advance(0.0);
        }
        // Train for random waveforms.
        while *nav_class.interp_sample_fraction.read().unwrap() >= 1.0
            || *kv_class.interp_sample_fraction.read().unwrap() >= 1.0
            || *nav_class.sched_sample_fraction.read().unwrap() >= 1.0
            || *kv_class.sched_sample_fraction.read().unwrap() >= 1.0
        {
            let mut state = State::new(nav_class.clone(), kv_class.clone());
            let inputs: Vec<_> = (0..(1 + (rand::random::<usize>() % 3)))
                .map(|_| -> Box<dyn training_patterns::Waveform> {
                    // if rand::random() {
                    Box::new(training_patterns::SquareWave::new())
                    // } else {
                    //     Box::new(training_patterns::SinWave::new())
                    // }
                })
                .collect();
            let mut ticks = 0; // Measure time with an integer for accuracy.
            let time = |ticks| ticks as f64 * args.time_step;
            while time(ticks) < 2000.0 {
                state.advance(inputs.iter().map(|x| x.eval(time(ticks))).sum());
                ticks += 1;
            }
            eprintln!("input:\n{:?}", inputs);
            eprintln!("nav:\n{}", nav_class);
            eprintln!("kv:\n{}", kv_class);
            nav_class.scheduler.write().unwrap().oversleep_error.clear();
            nav_class
                .scheduler
                .write()
                .unwrap()
                .undersleep_factor
                .clear();
            kv_class.scheduler.write().unwrap().oversleep_error.clear();
            kv_class
                .scheduler
                .write()
                .unwrap()
                .undersleep_factor
                .clear();
        }
    }

    // Run the users simulation.
    let mut state = State::new(nav_class.clone(), kv_class.clone());
    let mut ticks = 0; // Measure time with an integer for accuracy.
    let time = |ticks| ticks as f64 * args.time_step;
    let mut end_time = 0.0;
    let mut performance = 0;
    let mut num_computes = 0;
    loop {
        // Parse a command entry for the current injection probe.
        let command_str = match std::io::stdin().lock().lines().next() {
            Some(qq) => qq.unwrap(),
            None => break,
        };
        let mut command_iter = command_str.split_whitespace();
        let time_period: f64 = command_iter.next().unwrap().parse().unwrap();
        let i_inject: f64 = command_iter.next().unwrap().parse().unwrap();
        //
        end_time += time_period;
        let start = std::time::Instant::now();
        while time(ticks) < end_time {
            if args.numeric {
                state.advance_numeric(i_inject);
            } else {
                if args.train {
                    // Disable the automatic double checking of results,
                    // otherwise it always returns exact values (no
                    // interpolation, perfect scheduling).
                    *nav_class.interp_sample_fraction.write().unwrap() = 0.0;
                    *kv_class.interp_sample_fraction.write().unwrap() = 0.0;
                    *nav_class.sched_sample_fraction.write().unwrap() = 0.0;
                    *kv_class.sched_sample_fraction.write().unwrap() = 0.0;
                }
                state.advance(i_inject);
                if state.nav.last_compute == 0 {
                    num_computes += 1;
                }
            }
            ticks += 1;
            println!("{} {}", time(ticks), state.voltage);
        }
        performance += start.elapsed().as_nanos();
    }
    if num_computes > 0 {
        eprintln!("Compute Load: {} %", num_computes as f64 / ticks as f64);
    }
    eprintln!("Performance: {} seconds.", performance as f64 / 1e9);
}
