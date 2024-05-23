import subprocess
import time
import matplotlib.pyplot as plt


def run_experiments():
    processors = [1, 2, 4, 8, 16, 24, 48]
    execution_times = {}

    for p in processors:
        start_time = time.time()
        subprocess.run(['mpiexec', '-n', str(p), 'python', 'main.py'], capture_output=True, text=True)
        execution_times[p] = time.time() - start_time
        # print(execution_times)
    return execution_times

def calculate_efficiency(execution_times):
    speedup = {p: execution_times[1] / execution_times[p] for p in execution_times}
    efficiency = {p: speedup[p] / p for p in speedup}
    return efficiency


def plot_efficiency(efficiency):
    processors = sorted(efficiency.keys())
    efficiency_values = [efficiency[p] for p in processors]

    plt.figure(figsize=(10, 6))
    plt.plot(processors, efficiency_values, marker='o')
    plt.xlabel('Number of Processors')
    plt.ylabel('Efficiency')
    plt.title('Efficiency vs. Number of Processors')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    execution_times = run_experiments()
    efficiency = calculate_efficiency(execution_times)
    plot_efficiency(efficiency)