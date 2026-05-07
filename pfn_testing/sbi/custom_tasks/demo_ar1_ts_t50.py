"""Quick smoke demo for the custom AR(1) T=50 task."""

from __future__ import annotations

import matplotlib.pyplot as plt

from pfn_testing.sbi.sbibm_utils import get_task


def main() -> None:
    task = get_task("ar1_ts_t50")
    print(f"dim_theta={task.dim_theta}, dim_x={task.dim_x}")

    x1 = task.get_observation(num_observation=1)
    print("observation[1] first 5 values:", x1[0, :5].numpy())

    ref = task.get_reference_posterior_samples(num_observation=1)
    print("ref posterior mean [rho, log_sigma]:", ref.mean(dim=0).numpy())

    plt.figure(figsize=(8, 3))
    plt.plot(x1[0].numpy(), lw=1.5)
    plt.title("AR(1) reference observation (obs=1)")
    plt.xlabel("t")
    plt.ylabel("y_t")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
