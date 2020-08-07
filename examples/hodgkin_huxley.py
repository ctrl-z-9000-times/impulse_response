import numpy as np
import subprocess
import tempfile
import matplotlib.pyplot as plt
import scipy.interpolate

def run_simulation(inject_pattern, args=""):
    data = tempfile.TemporaryFile('w+t')
    subprocess.run(("cargo run --release --example hodgkin_huxley -- " + args).split(),
        input=bytes("\n".join(str(period) + " " + str(current)
            for period, current in inject_pattern), "utf8"),
        stdout=data)
    data.seek(0)
    hh = np.array([[float(x) for x in entry.split()]
            for entry in data.read().strip().split('\n')],
            dtype=np.float64)
    time = hh[:, 0]
    voltage = hh[:, 1]
    return (time, voltage)

def plot_test_pattern():
    inject_pattern = [
        # Demonstrate efficiency during quiet periods.
        # (4000, 0),
        # Regular Test Pattern:
        (45, 0),
        (.5, 0.05),
        (9, 0),
        (1, 0.05),
        (9, 0),
        (1.5, 0.05),
        (9, 0),
        (2, 0.05),
        (20, 0),
        (3, 0.05),
        (8, 0),
        (3, 0.05),
        (8, 0),
        (3, 0.05),
        (8, 0),
        (3, 0.05),
        (8, 0),
        (3, 0.05),
        (8, 0),
        (3, 0.05),
        (8, 0),
        (3, 0.05),
        (15, 0),
        (40, 0.05),
        (20, 0),
    ]
    approx_time, approx_voltage = run_simulation(inject_pattern, "--time_step 10e-3 --train")
    exact_time, exact_voltage = run_simulation(inject_pattern, "--time_step 10e-3 --numeric")
    plot_error = False
    if plot_error:
        plt.subplot(2, 1, 1)
    plt.title("The Hodgkin Huxley Model")
    plt.ylabel("Millivolts")
    plt.xlabel("Milliseconds")
    plt.plot(approx_time, approx_voltage, 'r+',
            exact_time, exact_voltage, 'g+');
    t = 0
    for period, inject in inject_pattern:
        if inject != 0:
            plt.axvspan(t, t + period, color='#FFFF00');
        t += period
    # Plot the absolute error.
    if plot_error:
        plt.subplot(2, 1, 2)
        plt.title("Absolute Error")
        appox_interp = scipy.interpolate.interp1d(approx_time, approx_voltage, fill_value="extrapolate")
        abs_error = [abs(v - appox_interp(t)) for v, t in zip(exact_voltage, exact_time)]
        plt.plot(exact_time, abs_error)
    plt.show()

if __name__ == "__main__":
    plot_test_pattern()
