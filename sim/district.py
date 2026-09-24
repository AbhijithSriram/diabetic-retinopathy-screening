"""Discrete-event simulation of a district DR screening programme (SimPy).

Mirrors matlab/build_simulink_model.m (SimEvents) and matlab/district_sim.m
(plain MATLAB): same entities, same parameters (sim/params.yaml).

Patient journey:
  arrive at a site -> photo attempts (quality gate, <= 3) -> on-device AI triage
  -> recall | review | refer   (+ a random audit sample of recalls goes to review)
  review/audit images wait for connectivity, upload, then join the tele-review
  queue; ophthalmologists review in daily windows at ~30 s per case
  referred patients attend with probability p_complete and then queue for
  hospital slots.

Answers the PS's resource-allocation question: how many reviewer-hours,
how much connectivity, and what hospital capacity a 100k/year district
needs, with and without AI triage and the referral-completion engine.

    python -m sim.district --sweep
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import simpy
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


SCENARIOS = {
    # name: (which evaluation block supplies the conditional triage rates)
    "calibrated camera": ("DDR test (internal)", "triage_cnn_only"),
    "new camera, frozen thresholds": ("IDRiD all 516 (external, India, Kowa)", "triage_cnn_only"),
    "new camera, recalibrated (200 audit labels)": ("IDRiD all 516 (external, India, Kowa)", "site_recalibrated_triage"),
}


def load_params(path=HERE / "params.yaml", eval_json: Path | None = None, scenario: str = "calibrated camera") -> dict:
    """Triage fractions come from measured conditional rates
    P(outcome | referable) and P(outcome | not referable), re-mixed to a
    screening population's prevalence (the datasets are far richer in
    disease than a real district)."""
    p = yaml.safe_load(Path(path).read_text())
    if eval_json and eval_json.exists():
        ev = json.loads(eval_json.read_text())
        key, block = SCENARIOS[scenario]
        t = (ev.get("recalibration", {}).get(key, {}).get(block) if block == "site_recalibrated_triage"
             else ev.get("sets", {}).get(key, {}).get(block))
        if t and t.get("recall_given_referable") is not None:
            pi = p["population"]["referable_prevalence"]
            mix = {o: pi * t[f"{o}_given_referable"] + (1 - pi) * t[f"{o}_given_non_referable"] for o in ("recall", "review", "refer")}
            p["triage"].update(mix)
            # share of reviewed cases that are truly referable (reviewer assumed accurate)
            p["review"]["refer_after_review"] = pi * t["review_given_referable"] / max(1e-9, mix["review"])
            p["triage"]["missed_referable_per_referable"] = t["recall_given_referable"]
            p["triage"]["true_referable_share_of_direct_refer"] = pi * t["refer_given_referable"] / max(1e-9, mix["refer"])
            p["triage"]["source"] = f"{eval_json.name}: {key} [{block}], re-mixed to prevalence {pi}"
            p["scenario"] = scenario
    return p


@dataclass
class Stats:
    arrivals: int = 0
    ungradable_referred: int = 0
    recall: int = 0
    review: int = 0
    audit: int = 0
    refer_direct: int = 0
    refer_after_review: int = 0
    attended: int = 0
    review_turnaround_h: list = field(default_factory=list)
    upload_wait_h: list = field(default_factory=list)
    hospital_wait_days: list = field(default_factory=list)
    reviewer_busy_h: float = 0.0
    reviewer_avail_h: float = 0.0
    capture_minutes: float = 0.0


class District:
    def __init__(self, p: dict, reviewers=None, online=None, completion=None, review_all=False, seed=None):
        self.p = p
        self.env = simpy.Environment()
        self.rng = np.random.default_rng(p["run"]["seed"] if seed is None else seed)
        self.n_rev = reviewers or p["review"]["reviewers"]
        self.online = p["network"]["online_fraction"] if online is None else online
        self.completion = p["referral"]["completion_baseline"] if completion is None else completion
        self.review_all = review_all
        self.s = Stats()
        d = p["district"]
        self.rate = d["annual_screenings"] / d["working_days_per_year"] / d["sites"] / d["site_hours_per_day"]  # per site-hour
        self.cameras = [simpy.Resource(self.env, 1) for _ in range(d["sites"])]
        self.uplinks = [simpy.Resource(self.env, 1) for _ in range(d["sites"])]
        self.link_up = [True] * d["sites"]
        self.link_event = [self.env.event() for _ in range(d["sites"])]
        self.review_q = simpy.Store(self.env)
        self.hospital_q = simpy.Store(self.env)

    # ------------------------------------------------------------- helpers
    def day_hour(self):
        return self.env.now % 24.0

    def tri(self, a, m, b):
        return self.rng.triangular(a, m, b)

    # ----------------------------------------------------------- processes
    def link(self, i):
        out_mean = self.p["network"]["outage_mean_hours"]
        up_mean = out_mean * self.online / max(1e-6, 1 - self.online) if self.online < 1 else 1e9
        while True:
            self.link_up[i] = True
            ev, self.link_event[i] = self.link_event[i], self.env.event()
            ev.succeed()
            yield self.env.timeout(self.rng.exponential(up_mean))
            if self.online >= 1:
                continue
            self.link_up[i] = False
            yield self.env.timeout(self.rng.exponential(out_mean))

    def arrivals(self, i):
        h = self.p["district"]["site_hours_per_day"]
        while True:
            hour = self.day_hour()
            if hour >= h:  # site closed: jump to next morning
                yield self.env.timeout(24.0 - hour)
                continue
            yield self.env.timeout(self.rng.exponential(1.0 / self.rate))
            if self.day_hour() < h:
                self.env.process(self.patient(i))

    def patient(self, i):
        s, p = self.s, self.p
        s.arrivals += 1
        t0 = self.env.now
        cap = p["capture"]
        ok = False
        intrinsic = self.rng.random() < cap["intrinsic_ungradable"]
        technique = self.rng.random() < cap["technique_failure"]
        for attempt in range(1, cap["max_attempts"] + 1):
            with self.cameras[i].request() as req:
                yield req
                m = self.tri(*cap["minutes_per_attempt"])
                s.capture_minutes += m
                yield self.env.timeout(m / 60.0)
            if attempt > 1 and technique and self.rng.random() < cap["retake_success"]:
                technique = False
            if not intrinsic and not technique:
                ok = True
                break
        if not ok:
            s.ungradable_referred += 1
            yield from self.referral()
            return
        tr = p["triage"]
        if self.review_all:
            outcome = "review"
        else:
            u = self.rng.random()
            outcome = "recall" if u < tr["recall"] else "review" if u < tr["recall"] + tr["review"] else "refer"
        if outcome == "recall":
            s.recall += 1
            if self.rng.random() < tr["audit_fraction_of_recalls"]:
                s.audit += 1
                self.env.process(self.upload_then_review(i, t0, audit=True))
            return
        if outcome == "refer":
            s.refer_direct += 1
            yield from self.referral()
            return
        s.review += 1
        yield from self.upload_then_review(i, t0, audit=False)

    def upload_then_review(self, i, t0, audit):
        net = self.p["network"]
        t_q = self.env.now
        with self.uplinks[i].request() as req:
            yield req
            while not self.link_up[i]:
                yield self.link_event[i]
            secs = net["image_mb"] * net["eyes_per_patient"] * 8 * 1024 / net["bandwidth_kbps"]
            yield self.env.timeout(secs / 3600.0)
        self.s.upload_wait_h.append(self.env.now - t_q)
        done = self.env.event()
        yield self.review_q.put((t0, done, audit))
        decision = yield done
        if not audit:
            self.s.review_turnaround_h.append(self.env.now - t0)
            if decision == "refer":
                self.s.refer_after_review += 1
                yield from self.referral()

    def reviewer(self, k):
        r = self.p["review"]
        start = 14.0 + 0.5 * k  # afternoon review windows, slightly staggered
        while True:
            hour = self.day_hour()
            wait = (start - hour) % 24.0
            yield self.env.timeout(wait)
            end = self.env.now + r["review_hours_per_reviewer_day"]
            self.s.reviewer_avail_h += r["review_hours_per_reviewer_day"]
            while self.env.now < end:
                if not self.review_q.items:
                    yield self.env.timeout(min(0.05, end - self.env.now))
                    continue
                t0, done, audit = yield self.review_q.get()
                dt = r["seconds_per_case"] / 3600.0
                yield self.env.timeout(dt)
                self.s.reviewer_busy_h += dt
                done.succeed("refer" if self.rng.random() < r["refer_after_review"] else "recall")
            yield self.env.timeout(0.01)

    def referral(self):
        if self.rng.random() < self.completion:
            self.s.attended += 1
            yield self.hospital_q.put(self.env.now)

    def hospital(self):
        slots = self.p["referral"]["hospital_slots_per_day"]
        while True:
            yield self.env.timeout(24.0 - self.day_hour() + 9.0 if self.day_hour() > 9 else 9.0 - self.day_hour())
            for _ in range(slots):
                if not self.hospital_q.items:
                    break
                t = yield self.hospital_q.get()
                self.s.hospital_wait_days.append((self.env.now - t) / 24.0)
            yield self.env.timeout(1.0)

    def run(self):
        d = self.p["district"]
        for i in range(d["sites"]):
            self.env.process(self.arrivals(i))
            self.env.process(self.link(i))
        for k in range(self.n_rev):
            self.env.process(self.reviewer(k))
        self.env.process(self.hospital())
        days = self.p["run"]["days"]
        self.env.run(until=days * 24.0)
        return self.summary(days)

    def summary(self, days):
        s = self.s
        yr = self.p["district"]["working_days_per_year"] / days
        tat = np.array(s.review_turnaround_h) if s.review_turnaround_h else np.array([np.nan])
        sla = self.p["review"]["sla_hours"]
        # cases still waiting at the end count as SLA misses if already older than the SLA
        pending = [self.env.now - t0 for t0, _, a in self.review_q.items if not a]
        n_done = len(s.review_turnaround_h)
        within = int((tat <= sla).sum()) if n_done else 0
        late_pending = sum(1 for w in pending if w > sla)
        hw = np.array(s.hospital_wait_days) if s.hospital_wait_days else np.array([0.0])
        return {
            "reviewers": self.n_rev, "online_fraction": self.online, "completion": self.completion, "review_all": self.review_all,
            "screened_per_year": round(s.arrivals * yr),
            "reviews_per_year": round((s.review + s.audit) * yr),
            "review_within_sla_pct": round(100 * within / max(1, n_done + late_pending), 1),
            "review_turnaround_median_h": round(float(np.nanmedian(tat)), 1),
            "review_turnaround_p95_h": round(float(np.nanpercentile(tat, 95)), 1),
            "review_backlog_end": len(self.review_q.items),
            "reviewer_utilisation_pct": round(100 * s.reviewer_busy_h / max(1e-9, s.reviewer_avail_h), 1),
            "upload_wait_p95_h": round(float(np.percentile(s.upload_wait_h, 95)), 2) if s.upload_wait_h else 0.0,
            "referrals_per_year": round((s.refer_direct + s.refer_after_review + s.ungradable_referred) * yr),
            "ungradable_referrals_per_year": round(s.ungradable_referred * yr),
            "attended_per_year": round(s.attended * yr),
            "hospital_wait_median_days": round(float(np.median(hw)), 1),
            "hospital_backlog_end": len(self.hospital_q.items),
        }


def scenario_table(eval_json: Path, out_dir: Path):
    rows = []
    for sc in SCENARIOS:
        p = load_params(eval_json=eval_json, scenario=sc)
        if p.get("scenario") != sc:
            continue
        for c in (0.15, 0.40):
            r = District(p, completion=c).run()
            tr = p["triage"]
            rows.append({"scenario": sc, "completion": c, "recall_pct": round(100 * tr["recall"], 1), "review_pct": round(100 * tr["review"], 1),
                         "direct_refer_pct": round(100 * tr["refer"], 1),
                         "true_referable_share_of_direct_refer": round(tr["true_referable_share_of_direct_refer"], 3),
                         "missed_referable_per_referable": tr["missed_referable_per_referable"],
                         **{k: r[k] for k in ("reviews_per_year", "referrals_per_year", "attended_per_year", "hospital_backlog_end", "hospital_wait_median_days")}})
    (out_dir / "sim_scenarios.json").write_text(json.dumps(rows, indent=2))
    return rows


def sweep(p, out_dir: Path):
    import csv
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for review_all in (True, False):
        for online in (0.3, 0.6, 0.9):
            for n in range(1, 9):
                r = District(p, reviewers=n, online=online, review_all=review_all).run()
                rows.append(r)
                print(json.dumps(r), flush=True)
    comp = []
    for c in (0.15, 0.25, 0.40, 0.60):
        comp.append(District(p, completion=c).run())
    with open(out_dir / "sim_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows + comp)
    # minimal reviewers meeting 95% within SLA
    best = {}
    for r in rows:
        key = f"{'all images reviewed' if r['review_all'] else 'AI triage'} | online {r['online_fraction']}"
        if r["review_within_sla_pct"] >= 95 and (key not in best or r["reviewers"] < best[key]["reviewers"]):
            best[key] = r
    summary = {"params": p, "min_reviewers_for_95pct_sla": {k: v["reviewers"] for k, v in best.items()},
               "completion_sweep": comp, "grid": rows}
    (out_dir / "sim_summary.json").write_text(json.dumps(summary, indent=2))
    _plots(rows, comp, out_dir)
    return summary


def _plots(rows, comp, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for review_all, ls in ((True, "--"), (False, "-")):
        for online, c in zip((0.3, 0.6, 0.9), ("#b45309", "#2563eb", "#059669")):
            rr = [r for r in rows if r["review_all"] == review_all and r["online_fraction"] == online]
            ax[0].plot([r["reviewers"] for r in rr], [r["review_within_sla_pct"] for r in rr], ls, color=c,
                       label=f"{'review all' if review_all else 'AI triage'}, online {int(online * 100)}%")
    ax[0].axhline(95, color="grey", lw=0.8)
    ax[0].set_xlabel("ophthalmologists doing tele-review (2 h/day each)")
    ax[0].set_ylabel("% reviewed within 48 h")
    ax[0].legend(fontsize=7)
    ax[0].set_title("Review capacity needed, 100k screenings/year")
    ax[1].bar([f"{int(c['completion'] * 100)}%" for c in comp], [c["attended_per_year"] for c in comp], color="#2563eb")
    ax[1].set_xlabel("referral completion rate")
    ax[1].set_ylabel("patients reaching the eye hospital / year")
    ax[1].set_title("What closing the referral gap is worth")
    fig.tight_layout()
    fig.savefig(out_dir / "sim_sweep.png", dpi=130)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--eval", default="", help="results/eval_<grader>.json to take triage fractions from")
    a = ap.parse_args()
    ev = REPO / "results" / a.eval if a.eval else None
    params = load_params(eval_json=ev)
    if a.sweep:
        out = REPO / "results" / "simulation"
        sweep(params, out)
        if ev:
            for r in scenario_table(ev, out):
                print(json.dumps(r))
    else:
        print(json.dumps(District(params).run(), indent=2))
