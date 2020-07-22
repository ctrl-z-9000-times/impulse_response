type Array = [f64; 4];

/// Maximum value of each bin.
const BINS: Array = [1.0, 10.0, 100.0, f64::INFINITY];

// TODO: Consider splitting the business logic (max/within-tolerance) apart from
// the histogram measurment code.

/// Maximum fraction of the histogram allowed in each bin.
/// If any of the bins excedes their threshold then the scheduler will be disabled.
const MAX: Array = [f64::NAN, 1e-3, 1e-4, 1e-6];

#[derive(Debug, Default)]
pub struct OversleepErrorHistogram {
    data: Array,
}

impl OversleepErrorHistogram {
    pub fn decay(&mut self, decay_factor: f64) {
        debug_assert!(decay_factor >= 0.0 && decay_factor <= 1.0);
        for bin in self.data.iter_mut() {
            *bin *= decay_factor
        }
    }

    pub fn add(&mut self, error: f64) {
        debug_assert!(error >= 0.0);
        debug_assert!(!error.is_nan());
        for bin in 0..BINS.len() {
            if error <= BINS[bin] {
                self.data[bin] += 1.0;
                break;
            }
        }
    }

    fn normalize(&self) -> Array {
        let mut x = self.data.clone();
        let sum: f64 = x.iter().sum();
        x.iter_mut().for_each(|v| *v /= sum);
        x
    }

    pub fn within_tolerance(&self) -> bool {
        self.normalize()
            .iter()
            .skip(1)
            .zip(MAX.iter().skip(1))
            .all(|(bin, max_pct)| bin <= max_pct)
    }
}

impl std::fmt::Display for OversleepErrorHistogram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Oversleep Error Histogram:\n")?;
        let hist = self.normalize();
        let mut lower = 0.0;
        for (upper, pct) in BINS.iter().zip(hist.iter()) {
            if lower == 0.0 {
                write!(f, "  [{:3}, {:3}]: {:7.3} %\n", lower, upper, pct * 100.0)?;
            } else {
                write!(f, "  ({:3}, {:3}]: {:7.3} %\n", lower, upper, pct * 100.0)?;
            }
            lower = *upper;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sane_bins() {
        assert!(BINS[0] == 1.0);
        assert!(BINS[BINS.len() - 1] == f64::INFINITY);
        for idx in 0..BINS.len() - 1 {
            assert!(BINS[idx] < BINS[idx + 1])
        }
    }

    #[test]
    fn sane_max() {
        assert!(MAX[0].is_nan());
        assert!(MAX.iter().skip(1).sum::<f64>() <= 1.0);
        assert!(MAX.iter().skip(1).all(|&m| m >= 0.0));
        for idx in 1..MAX.len() - 1 {
            assert!(MAX[idx] >= MAX[idx + 1])
        }
    }

    #[test]
    fn unit_test() {
        let mut h = OversleepErrorHistogram::default();
        println!("{}", h);
        assert!(h.within_tolerance() == false);
        for _ in 0..1000 {
            h.add(rand::random());
        }
        println!("{}", h);
        assert!(h.within_tolerance() == true);
        for i in 1..10 {
            h.add(i as f64)
        }
        println!("{}", h);
        assert!(h.within_tolerance() == false);
        for _ in 0..1_000_000 {
            h.add(rand::random());
        }
        println!("{}", h);
        assert!(h.within_tolerance() == true);
    }
}
