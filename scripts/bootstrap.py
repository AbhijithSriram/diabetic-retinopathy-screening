"""One-command setup for a new machine.

    python scripts/bootstrap.py                  # everything: packages, models, MATLAB engine, self-check
    python scripts/bootstrap.py --backend python # skip MATLAB
    python scripts/bootstrap.py --models-only    # just (re)fetch the models

1. installs requirements.txt (unless --skip-install);
2. downloads the model release pinned in configs/model_release.json, checks
   each SHA-256 and unpacks it into runs/ and app/samples/; files already
   present and unchanged are not downloaded again;
3. finds MATLAB, installs the MATLAB Engine for Python that ships with it
   (also without administrator rights, which the plain pip install needs on
   Windows) and checks the toolboxes the app uses;
4. screens a sample photograph on each engine and prints how to start the app.

The repository is private, so downloading the release needs GitHub access:
either the GitHub CLI logged in (`gh auth login`) or a token with read access
in the GITHUB_TOKEN environment variable.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RELEASE_CFG = REPO / "configs" / "model_release.json"
STAMP = REPO / "runs" / ".release.json"  # which assets are unpacked, by hash
TOOLBOXES = {"Neural_Network_Toolbox": "Deep Learning Toolbox", "Image_Toolbox": "Image Processing Toolbox"}


def say(msg=""):
    print(msg, flush=True)


def step(title):
    say(f"\n== {title}")


def fail(msg, code=1):
    say(f"\nSTOPPED: {msg}")
    sys.exit(code)


# ------------------------------------------------------------------ packages
def install_packages():
    step("Python packages")
    v = sys.version_info
    if not (3, 10) <= (v.major, v.minor) <= (3, 13):
        say(f"  warning: Python {v.major}.{v.minor}; tested with 3.10 to 3.13")
    cmd = [sys.executable, "-m", "pip", "install", "-r", str(REPO / "requirements.txt")]
    if sys.prefix == sys.base_prefix and not os.environ.get("CONDA_PREFIX"):
        cmd.insert(4, "--user")  # no virtual environment: do not write into the system Python
    say("  " + " ".join(cmd))
    if subprocess.call(cmd) != 0:
        fail("pip install failed (see above).")
    import torch
    say(f"  torch {torch.__version__}; GPU: {'yes, ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no (the CPU is used; slower but fine)'}")


# ------------------------------------------------------------------ models
def _gh_token():
    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if tok:
        return tok
    gh = shutil.which("gh") or next(iter(glob.glob(r"C:\Program Files\GitHub CLI\gh.exe")), None)
    if gh:
        try:
            out = subprocess.run([gh, "auth", "token"], capture_output=True, text=True, timeout=20)
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        except Exception:
            pass
    return None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


def _download(repo, tag, name, dest: Path, token):
    api = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
    hdr = {"Accept": "application/vnd.github+json", "User-Agent": "drishtitrust-bootstrap"}
    if token:
        hdr["Authorization"] = f"Bearer {token}"
    try:
        rel = json.load(urllib.request.urlopen(urllib.request.Request(api, headers=hdr), timeout=30))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            fail(f"release {tag} not found in {repo}. The repository is private: log in with `gh auth login` "
                 "(or set GITHUB_TOKEN), and make sure your GitHub account has been added to the repository.")
        raise
    asset = next((x for x in rel["assets"] if x["name"] == name), None)
    if asset is None:
        fail(f"release {tag} has no asset {name}")
    # the asset API answers with a redirect to signed storage; the token must
    # not be sent there, so the redirect is followed by hand
    req = urllib.request.Request(asset["url"], headers={**hdr, "Accept": "application/octet-stream"})
    try:
        r = urllib.request.build_opener(_NoRedirect).open(req, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code not in (301, 302, 303, 307, 308):
            raise
        r = urllib.request.urlopen(urllib.request.Request(e.headers["Location"], headers={"User-Agent": hdr["User-Agent"]}), timeout=60)
    total = int(r.headers.get("Content-Length") or asset["size"])
    h = hashlib.sha256()
    done, t0, last = 0, time.time(), 0.0
    with open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
            h.update(chunk)
            done += len(chunk)
            if time.time() - last > 1 or done == total:
                last = time.time()
                rate = done / max(1e-3, last - t0) / 1e6
                print(f"\r  {name}: {done / 1e6:6.0f} / {total / 1e6:.0f} MB  ({rate:.1f} MB/s)   ", end="", flush=True)
    print()
    return h.hexdigest()


def fetch_models(backend):
    step("Model files")
    cfg = json.loads(RELEASE_CFG.read_text())
    stamp = json.loads(STAMP.read_text()) if STAMP.exists() else {}
    want = {n: a for n, a in cfg["assets"].items() if a["for"] in ("all", backend) or backend == "both"}
    todo = {n: a for n, a in want.items() if stamp.get(n) != a["sha256"] or not _unpacked(n)}
    for n in want:
        if n not in todo:
            say(f"  {n}: already in place")
    if not todo:
        return
    token = _gh_token()
    if not token:
        say("  no GitHub login found: trying without one (works only if the repository is public)")
    free = shutil.disk_usage(REPO).free
    need = sum(2 * a["bytes"] for a in todo.values())
    if free < need:
        fail(f"not enough disk space: {need / 1e9:.1f} GB needed, {free / 1e9:.1f} GB free")
    with tempfile.TemporaryDirectory() as d:
        for n, a in todo.items():
            p = Path(d) / n
            got = _download(cfg["repo"], cfg["tag"], n, p, token)
            if got != a["sha256"]:
                fail(f"{n}: checksum mismatch (got {got[:12]}, expected {a['sha256'][:12]}); the download is damaged or the release changed")
            with zipfile.ZipFile(p) as z:
                for m in z.namelist():  # every path must stay inside the repository
                    if Path(m).is_absolute() or ".." in Path(m).parts:
                        fail(f"{n}: unsafe path {m}")
                z.extractall(REPO)
            stamp[n] = a["sha256"]
            STAMP.parent.mkdir(exist_ok=True)
            STAMP.write_text(json.dumps(stamp, indent=1))
            say(f"  {n}: verified and unpacked")


def _unpacked(name):
    probe = {"drishtitrust_models_pytorch.zip": "runs/v2s_d2/best.pt",
             "drishtitrust_models_matlab.zip": "runs/onnx_matlab/manifest.json",
             "drishtitrust_samples.zip": "app/samples/hrf_01_dr.jpg"}.get(name)
    return probe is None or (REPO / probe).exists()


# ------------------------------------------------------------------ MATLAB
def find_matlab() -> Path | None:
    if os.environ.get("MATLAB_ROOT"):
        return Path(os.environ["MATLAB_ROOT"])
    exe = shutil.which("matlab")
    if exe:
        return Path(exe).resolve().parent.parent
    pats = {"Windows": [r"C:\Program Files\MATLAB\R20*"], "Darwin": ["/Applications/MATLAB_R20*.app"],
            "Linux": ["/usr/local/MATLAB/R20*", "/opt/MATLAB/R20*", os.path.expanduser("~/MATLAB/R20*")]}
    roots = sorted(p for pat in pats.get(platform.system(), []) for p in glob.glob(pat))
    return Path(roots[-1]) if roots else None  # newest release


def _engine_ok():
    return subprocess.call([sys.executable, "-c", "import matlab.engine"], stderr=subprocess.DEVNULL) == 0


def install_engine(root: Path):
    src = root / "extern" / "engines" / "python"
    if not src.exists():
        fail(f"{src} not found; is {root} a full MATLAB installation?")
    say(f"  installing the MATLAB Engine for Python from {src}")
    if subprocess.call([sys.executable, "-m", "pip", "install", str(src)]) == 0 and _engine_ok():
        return
    # Without write access to the MATLAB folder pip cannot build the engine
    # there, and its setup.py finds MATLAB relative to its own folder, so it
    # cannot be built from a plain copy either. Build it in a mirror that links
    # back to MATLAB's bin folders, then point the installed package at the
    # real MATLAB.
    say("  plain install failed (usually no administrator rights); building from a linked copy instead")
    with tempfile.TemporaryDirectory() as d:
        mirror = Path(d) / "mlroot"
        shutil.copytree(src, mirror / "extern" / "engines" / "python")
        for rel in ("bin", "extern/bin"):
            link, target = mirror / rel, root / rel
            if platform.system() == "Windows":
                subprocess.check_call(["cmd", "/c", "mklink", "/J", str(link), str(target)], stdout=subprocess.DEVNULL)
            else:
                link.symlink_to(target, target_is_directory=True)
        if subprocess.call([sys.executable, "-m", "pip", "install", "--user", "--no-build-isolation", "."],
                           cwd=mirror / "extern" / "engines" / "python") != 0:
            fail("could not build the MATLAB Engine for Python; ask for help with the output above")
        for rel in ("bin", "extern/bin"):  # remove the links, never the targets
            link = mirror / rel
            os.rmdir(link) if platform.system() == "Windows" else link.unlink()
    code = ("import importlib.util, os, sys; loc = importlib.util.find_spec('matlab').submodule_search_locations[0];"
            "f = os.path.join(loc, 'engine', '_arch.txt'); arch = open(f).read().split()[0]; r = sys.argv[1];"
            "open(f, 'w').write(os.linesep.join([arch, os.path.join(r, 'bin', arch),"
            " os.path.join(r, 'extern', 'engines', 'python', 'dist', 'matlab', 'engine', arch),"
            " os.path.join(r, 'extern', 'bin', arch)]) + os.linesep)")
    subprocess.check_call([sys.executable, "-c", code, str(root)])
    if not _engine_ok():
        fail("the MATLAB Engine for Python installed but does not import")


def check_matlab():
    step("MATLAB")
    root = find_matlab()
    if root is None:
        say("  MATLAB not found (set MATLAB_ROOT if it is installed somewhere unusual); the app will use PyTorch")
        return False
    say(f"  found {root}")
    if not _engine_ok():
        install_engine(root)
    say("  starting MATLAB (takes a while the first time)...")
    import matlab.engine
    eng = matlab.engine.start_matlab()
    try:
        say(f"  MATLAB {eng.version()}")
        missing = [name for key, name in TOOLBOXES.items() if not eng.license("test", key)]
        if not eng.exist("importNetworkFromONNX"):
            missing.append("Deep Learning Toolbox Converter for ONNX Model Format (Home > Add-Ons)")
        if missing:
            fail("MATLAB is missing: " + "; ".join(missing))
        say(f"  GPU in MATLAB: {'yes' if eng.eval('canUseGPU()') else 'no (the CPU is used)'}")
        eng.cd(str(REPO / "matlab"), nargout=0)
        say("  parity check against the Python models (check_export)...")
        eng.eval("T = check_export();", nargout=0)
    finally:
        eng.quit()
    return True


# ------------------------------------------------------------------ self-check
def self_check(backends):
    step("Self-check: screening a sample photograph")
    code = ("import sys, time, cv2; sys.path.insert(0, '.'); b = sys.argv[1];"
            "from drishti.pipeline import Screener; from drishti.matlab_backend import MatlabScreener;"
            "s = Screener() if b == 'python' else MatlabScreener();"
            "rgb = cv2.cvtColor(cv2.imread('app/samples/idrid_test_grade2_009.jpg'), cv2.COLOR_BGR2RGB);"
            "t = time.time(); r = s.screen(rgb, with_images=False);"
            "print(f'  {b}: outcome {r[\"outcome\"]}, P(referable) {r[\"grade\"][\"p_referable\"]:.3f}, missing {s.missing}, {time.time() - t:.1f} s')")
    ok = True
    for b in backends:
        if subprocess.call([sys.executable, "-c", code, b], cwd=REPO) != 0:
            ok = False
            say(f"  {b}: FAILED (see above)")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["both", "python", "matlab"], default="both")
    ap.add_argument("--skip-install", action="store_true", help="do not pip install requirements.txt")
    ap.add_argument("--models-only", action="store_true", help="only download the models")
    a = ap.parse_args()
    if a.models_only:
        fetch_models(a.backend)
        return
    if not a.skip_install:
        install_packages()
    fetch_models(a.backend)
    backends = []
    if a.backend in ("both", "python"):
        backends.append("python")
    if a.backend in ("both", "matlab") and check_matlab():
        backends.append("matlab")
    ok = self_check(backends)
    step("Done" if ok else "Finished with errors")
    say("  start the app:   python -m uvicorn app.main:app --port 8026")
    say("  then open:       http://127.0.0.1:8026")
    if "matlab" in backends:
        say("  MATLAB instead:  set DRISHTI_BACKEND=matlab first (PowerShell: $env:DRISHTI_BACKEND='matlab')")
    say("  full UI test:    python scripts/e2e_ui.py")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
