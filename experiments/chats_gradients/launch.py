"""
Run a list of commands (one per line of a file) several at a time, each bound to its own
processor cores, each with its own log; a command that fails is run once more at the end.

    python launch.py TASKS.txt LOGDIR [--cores 1]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("tasks")
ap.add_argument("logs")
ap.add_argument("--cores", type=int, default=1, help="cores per command")
args = ap.parse_args()

tasks = [t.strip() for t in Path(args.tasks).read_text().splitlines() if t.strip()]
logs = Path(args.logs)
logs.mkdir(parents=True, exist_ok=True)
cores = sorted(os.sched_getaffinity(0))
cores = cores[: int(os.environ.get("NPAR", len(cores)))]  # what the job asked for
slots = [cores[a : a + args.cores] for a in range(0, len(cores) - args.cores + 1, args.cores)]
print(len(tasks), "commands,", len(slots), "at a time", flush=True)

queue = [(i, t, 1) for i, t in enumerate(tasks)]
running, free, failed = {}, list(range(len(slots))), []
while queue or running:
    while queue and free:
        i, task, attempt = queue.pop(0)
        slot = free.pop()
        log = (logs / f"task_{i:03d}.log").open("a")
        log.write(f"\n### attempt {attempt}: {task}\n")
        log.flush()
        cmd = f"taskset -c {','.join(map(str, slots[slot]))} {task}"
        p = subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT)
        running[p] = (i, task, attempt, slot, time.time())
    time.sleep(5)
    for p in [p for p in running if p.poll() is not None]:
        i, task, attempt, slot, t0 = running.pop(p)
        free.append(slot)
        status = "ok" if p.returncode == 0 else f"FAILED ({p.returncode})"
        print(f"{status} {time.time() - t0:6.0f} s  [{i}] {task}", flush=True)
        if p.returncode != 0:
            if attempt == 1:
                queue.append((i, task, 2))
            else:
                failed.append(task)
print("failed twice:", failed or "none")
