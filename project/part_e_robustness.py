"""
PART E - Integration and robustness.

Run:  python part_e_robustness.py           (about 6 minutes)
      python part_e_robustness.py --quick   (fewer trials, for a quick look)

Everything from Parts A to D runs together here, over repeated trials with the
block in a different place every time. Then we start breaking things on
purpose to find out where it stops working.

Writes: results/part_e_*.png
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import trackb as tb

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)

QUICK = "--quick" in sys.argv
N_TRIALS = 6 if QUICK else 15


def run_condition(name, n=N_TRIALS, **kwargs):
    """Run the same n block placements under one condition.

    Seeds 0..n-1 are reused for every condition, so a difference between two
    rows is the condition itself and not luck about where the block landed.
    """
    results = []
    for seed in range(n):
        try:
            ok, info, _ = tb.run_pick(seed=seed, **kwargs)
        except Exception as exc:                 # a crash counts as a failure
            ok, info = False, {"reason": f"error: {type(exc).__name__}"}
        results.append((ok, info))
    n_ok = sum(ok for ok, _ in results)

    def mean_of(key):
        values = [i[key] for ok, i in results if ok and key in i]
        return float(np.mean(values)) if values else float("nan")

    reasons = {}
    for ok, info in results:
        if not ok:
            reasons[info["reason"]] = reasons.get(info["reason"], 0) + 1

    return {"name": name, "n": n, "ok": n_ok,
            "grasp": mean_of("grasp_error_mm"), "object": mean_of("object_error_mm"),
            "track": mean_of("tracking_rms_mrad"), "time": mean_of("time_s"),
            "failures": reasons, "raw": results}


def show(row):
    fail = ", ".join(f"{k} x{v}" for k, v in row["failures"].items()) or "-"
    print(f"  {row['name']:<28} {row['ok']:>2}/{row['n']}   {row['grasp']:7.2f} "
          f"{row['object']:8.2f} {row['track']:8.2f}   {fail}")


print("=" * 78)
print(f"PART E - {N_TRIALS} trials per condition, block in a new place each time")
print("=" * 78)
print("\n  condition                   success  grasp[mm] object[mm] track[mrad]  failures")
print("  " + "-" * 74)

rows = []

# The nominal case.
nominal = run_condition("nominal")
rows.append(nominal)
show(nominal)

# More sensor noise: multiplies encoder, depth and image noise together.
for scale in [3, 6, 10]:
    row = run_condition(f"sensor noise x{scale}", noise_scale=scale)
    rows.append(row)
    show(row)

# Something shoves the arm.
for force in [40, 120]:
    row = run_condition(f"push {force} N on forearm", push=(force, 0, 0))
    rows.append(row)
    show(row)

# The block gets knocked mid-approach.
for dx in [0.03, 0.06]:
    row = run_condition(f"block moved {dx*100:.0f} cm", nudge=(dx, -dx * 0.7))
    rows.append(row)
    show(row)

# An unmodelled weight on the gripper.
for mass in [0.5, 2.0]:
    row = run_condition(f"unmodelled payload {mass} kg", payload=mass)
    rows.append(row)
    show(row)

print()


print("=" * 78)
print("WHAT THE VISION AND THE FILTER ACTUALLY BUY")
print("=" * 78)
print("""
Two claims worth testing rather than asserting.

"Open loop" means the robot takes its first look, then switches the camera off
and drives to wherever it thought the block was.

"No velocity feedforward" keeps the camera and the filter, and only stops the
controller using the velocity estimate - the arm chases the block's position
instead of travelling with it.

Read the two open-loop rows as separate experiments, not as a pair: a parked
block is drawn from the whole table, a moving one always starts at the upstream
edge, so the placements differ and the success rates are not comparable
like-for-like.
""")
print("  condition                   success  grasp[mm] object[mm] track[mrad]  failures")
print("  " + "-" * 74)

ablation = [
    run_condition("parked block, open loop", open_loop=True, block_speed=0.0),
    run_condition("moving block, open loop", open_loop=True),
    run_condition("no velocity feedforward", use_v_feedforward=False),
    run_condition("closed loop, block moved", nudge=(0.04, -0.03)),
    run_condition("open loop, block moved", open_loop=True, nudge=(0.04, -0.03)),
]
for row in ablation:
    show(row)

parked_ol, moving_ol, ff = ablation[0], ablation[1], ablation[2]
print(f"""
  Open loop: {parked_ol['ok']}/{parked_ol['n']} with the block parked and {moving_ol['ok']}/{moving_ol['n']} with it moving,
  against {nominal['ok']}/{nominal['n']} closed loop on the same placements. The reason is the
  filter itself. A constant-velocity filter with nothing to correct it does not
  sit still - it EXTRAPOLATES, along whatever velocity it happened to infer
  from its first three frames. Guess 10 mm/s wrong and six seconds of approach
  turns that into 60 mm of error. The old constant-position filter had no
  velocity to be wrong about, so it degraded more gracefully.

  That is the honest price of the motion model: it is what lets the robot
  coast through the blind frames at the end of the descent, and it is what
  makes running without measurements dangerous. A motion model is only worth
  having while something keeps correcting it.

  Velocity feedforward on its own:  {ff['ok']}/{ff['n']} picked up without it.

  Nothing is broken in that run - the camera is running, the filter is tracking
  the block to a few mm, the arm is healthy. It simply never closes. A
  proportional law settles at error = speed/gain, roughly 14 mm at belt speed,
  and the grasp will not fire until the error is under 3 mm, so the tool trails
  the block across the table until the phase times out. It is the plainest
  demonstration in this project that estimating a state and USING it are two
  different things.""")

closed, opened = ablation[3], ablation[4]
print(f"""
  With the block knocked 4 cm after the robot has locked on:
      camera still running   {closed['ok']}/{closed['n']} picked up
      camera switched off    {opened['ok']}/{opened['n']} picked up

  That gap is the whole point of visual servoing. With the camera running, the
  object filter sees several measurements that disagree with its estimate, its
  gate rejects them, and after four in a row it concludes the block has really
  moved and re-initialises there. Open loop there is nothing to notice with, so
  the arm closes its gripper on empty table.""")


all_rows = rows + ablation

fig, ax = plt.subplots(figsize=(11, 4))
names = [r["name"] for r in all_rows]
rates = [100 * r["ok"] / r["n"] for r in all_rows]
colours = ["tab:green" if r >= 90 else "tab:orange" if r >= 50 else "tab:red" for r in rates]
ax.bar(range(len(names)), rates, color=colours, width=0.7)
for i, (rate, row) in enumerate(zip(rates, all_rows)):
    ax.text(i, rate + 2, f"{row['ok']}/{row['n']}", ha="center", fontsize=7)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=38, ha="right", fontsize=8)
ax.set_ylabel("picks completed [%]")
ax.set_ylim(0, 112)
ax.set_title(f"Success rate by condition ({N_TRIALS} trials each, same block placements)",
             fontweight="bold")
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(RESULTS / "part_e_success.png", dpi=150)

# Spread of the nominal results.
fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
for ax, key, label in [(axes[0], "grasp_error_mm", "grasp error [mm]"),
                       (axes[1], "object_error_mm", "block estimate error [mm]"),
                       (axes[2], "tracking_rms_mrad", "joint tracking RMS [mrad]")]:
    values = [i[key] for ok, i in nominal["raw"] if ok and key in i]
    ax.hist(values, bins=8, color="tab:blue", alpha=0.85)
    ax.axvline(np.mean(values), color="tab:orange", lw=2,
               label=f"mean {np.mean(values):.2f}")
    ax.set_xlabel(label)
    ax.set_ylabel("trials")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
fig.suptitle(f"Nominal performance over {nominal['n']} randomised trials", fontweight="bold")
fig.tight_layout()
fig.savefig(RESULTS / "part_e_spread.png", dpi=150)

print(f"\nSaved two figures to {RESULTS}")


print()
print("=" * 78)
print("SUMMARY")
print("=" * 78)
grasp_values = [i["grasp_error_mm"] for ok, i in nominal["raw"] if ok]

# The verdict below is computed from the rows above, not written by hand. An
# earlier version of this script printed a fixed "survives everything" claim
# that its own table had stopped supporting.
noise_rows = [r for r in rows if r["name"].startswith("sensor noise")]
clean_noise = [r for r in noise_rows if r["ok"] == r["n"]]
broken_noise = [r for r in noise_rows if r["ok"] < r["n"]]

other_rows = [r for r in rows if r not in noise_rows and r["name"] != "nominal"]
clean_other = [r["name"] for r in other_rows if r["ok"] == r["n"]]
broken_other = [r for r in other_rows if r["ok"] < r["n"]]

if clean_noise:
    noise_line = (f"sensor noise up to x{clean_noise[-1]['name'].split('x')[-1]} "
                  f"with no failures")
else:
    noise_line = "no raised-noise condition at all"

if broken_noise:
    first = broken_noise[0]
    worst = max(first["failures"].items(), key=lambda kv: kv[1])[0]
    break_line = (f"Breaks first at {first['name']}: {first['ok']}/{first['n']} picked up, "
                  f"mostly \"{worst}\".")
else:
    break_line = "Nothing in the noise sweep broke it."

survive_line = "Survives " + noise_line
if clean_other:
    survive_line += ", plus " + ", ".join(clean_other)
survive_line += "."
if broken_other:
    survive_line += ("\n  Degraded but not clean: "
                     + ", ".join(f"{r['name']} ({r['ok']}/{r['n']})" for r in broken_other) + ".")

print(f"""
  Nominal success             {nominal['ok']}/{nominal['n']}
  Grasp error                 {np.mean(grasp_values):.2f} +- {np.std(grasp_values):.2f} mm
  Cycle time                  {nominal['time']:.1f} s

  {survive_line}
  {break_line}

  The limits worth stating plainly:
    * the detector is a colour threshold, so a red object on a red table, or
      strongly coloured lighting, would defeat it;
    * the grasp assumes a box lying flat on a table;
    * there is no collision checking on the way in - it works because the
      table is clear, not because anything is watching for obstacles.""")
