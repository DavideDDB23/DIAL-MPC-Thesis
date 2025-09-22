import matplotlib.pyplot as plt
import numpy as np
import os
import scienceplots

results = {
    "DIAL-MPC": {"reward": -3.39e-02, "jerk": 1.68e+02},
    "DIAL-MPC + Low Pass": {"reward": -4.56e-02, "jerk": 1.49e+02},
    "DIAL-MPC + Colored": {"reward": -4.42e-02, "jerk": 1.72e+02},
    "MPPI": {"reward": -2.01e-01, "jerk": 2.36e+02},
}

labels = list(results.keys())
rewards = [res["reward"] for res in results.values()]
jerks = [res["jerk"] for res in results.values()]

plt.rcParams.update({
    "text.usetex": False,
})

fig, ax = plt.subplots(figsize=(10, 6))

scatter = ax.scatter(jerks, rewards, s=100)
ax.set_xlabel("Control Variation - Lower is Better")
ax.set_ylabel("Mean Reward - Higher is Better")
ax.set_title("Performance vs. Control Smoothness Trade-off")
ax.grid(True)

for i, label in enumerate(labels):
    ax.text(jerks[i] + 1, rewards[i], label, fontsize=9)


output_dir = "results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

plt.savefig(os.path.join(output_dir, "performance_tradeoff.pdf"))
plt.show()

print(f"Plot saved to {os.path.join(output_dir, 'performance_tradeoff.pdf')}") 