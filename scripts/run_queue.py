"""Run GPU jobs one after another so the card is never idle and never
double-booked. One queue at a time; a job may itself be sized to share the card.

    python scripts/run_queue.py queue.json

The queue file is a list of {"name": ..., "cmd": [...], "needs": [paths]}.
A job waits until every path in `needs` exists, then runs; its output goes
to runs/<name>.log. A failing job does not stop the queue, because the later
jobs are usually independent candidates rather than continuations.

Only one queue may run at a time, enforced by a lock file. This is not
belt and braces: a queue replaced mid-flight on 20 Sep left its predecessor
alive, both were waiting on the same file, and when it appeared both
launched the same training into the same output directory. The two runs
raced over one checkpoint and one 8 GB card, and the only visible symptom
was a log file that stayed empty. The lock makes that failure loud.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = str(REPO / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))
LOCK = REPO / "runs" / ".queue.lock"


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def _alive(pid: int) -> bool:
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                             capture_output=True, text=True, timeout=20).stdout
        return str(pid) in out
    except Exception:
        return True  # if we cannot tell, assume it is and refuse to start


def take_lock():
    if LOCK.exists():
        try:
            held = json.loads(LOCK.read_text())
        except Exception:
            held = {}
        pid = int(held.get("pid", -1))
        if pid > 0 and _alive(pid):
            sys.exit(f"another queue is running (pid {pid}, {held.get('queue')}). "
                     f"Stop it first, or delete {LOCK} if you are sure it is dead.")
        log(f"clearing a stale lock from pid {pid}")
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(json.dumps({"pid": os.getpid(), "queue": sys.argv[1],
                                "started": datetime.now().isoformat(timespec="seconds")}))


def main():
    take_lock()
    queue = json.loads(Path(sys.argv[1]).read_text())
    for job in queue:
        name = job["name"]
        for need in job.get("needs", []):
            p = Path(need)
            waited = 0
            while not p.exists():
                if waited == 0:
                    log(f"{name}: waiting for {p.name}")
                time.sleep(30)
                waited += 30
                if waited > 7200:
                    log(f"{name}: gave up waiting for {p.name}")
                    break
        out = REPO / "runs" / f"{name}.log"
        # -u: without it Python buffers stdout when it is a file, so a job
        # that is running fine looks identical to one that hung at startup
        cmd = [PY, "-u"] + job["cmd"]
        log(f"{name}: {' '.join(job['cmd'])}")
        t = time.time()
        with out.open("w") as f:
            rc = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT).returncode
        log(f"{name}: exit {rc} after {(time.time() - t) / 60:.0f} min -> {out.name}")
    log("queue empty")
    LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
