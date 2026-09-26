# Setting up DrishtiTrust on a new machine

About 15 minutes of work plus downloads (about 800 MB of models, and more for
Python packages and MATLAB). A GPU is **not** needed: without one everything
runs on the CPU, just more slowly (timings below).

## 1. What you need

| | Required? | Notes |
|---|---|---|
| **Git** | yes | to clone the repository |
| **Python 3.10 to 3.13, 64-bit** | yes | from python.org or Miniconda. Tick "Add Python to PATH" on Windows. |
| **GitHub access to the repository** | yes | it is private: ask Abhijith to add your GitHub account |
| **GitHub CLI** (`gh`) | recommended | lets the setup script download the models with your login ([cli.github.com](https://cli.github.com)); a token in `GITHUB_TOKEN` works instead |
| **MATLAB R2024b or later** | optional | only to run the models in MATLAB; see [MATLAB](#4-matlab-optional) |
| **Disk space** | about 5 GB | packages about 3 GB, models about 0.8 GB (plus as much again while unpacking) |

## 2. Get the code and log in to GitHub

```bash
git clone https://github.com/AbhijithSriram/diabetic-retinopathy-screening.git
cd diabetic-retinopathy-screening
gh auth login
```

For `gh auth login`, choose GitHub.com, HTTPS and "Login with a web browser".

## 3. Run the setup script

```bash
python scripts/bootstrap.py
```

This one command:

1. installs the Python packages from `requirements.txt`;
2. downloads the trained models from the GitHub release named in
   `configs/model_release.json`, checks every file against its SHA-256 and
   unpacks it into `runs/` and `app/samples/`. Run it again any time: files
   that are already in place are not downloaded twice, and a new model release
   is picked up automatically;
3. if MATLAB is installed, installs the MATLAB Engine for Python and checks
   the toolboxes (skip with `--backend python`);
4. screens a sample photograph on each engine and prints how to start the app.

It ends with a line per engine such as
`python: outcome refer, P(referable) 0.926, missing [], 2.1 s`.
`missing []` means every model was found.

Useful variations:

```bash
python scripts/bootstrap.py --backend python   # no MATLAB
python scripts/bootstrap.py --models-only      # just download or update the models
python scripts/bootstrap.py --skip-install     # packages already installed
```

## 4. MATLAB (optional)

The app can run every network in MATLAB instead of PyTorch. That is the version
to demonstrate for the MathWorks problem statement.

1. Install MATLAB R2024b or later with **Deep Learning Toolbox** and **Image
   Processing Toolbox**. Parallel Computing Toolbox is only needed to use a
   GPU; without it MATLAB uses the CPU.
2. In MATLAB, go to **Home > Add-Ons**, and install **Deep Learning Toolbox
   Converter for ONNX Model Format**.
3. Run `python scripts/bootstrap.py` (again). It finds MATLAB in the usual
   install folder; if yours is elsewhere, set `MATLAB_ROOT` to the folder that
   contains `bin` first. The script installs the MATLAB Engine for Python,
   with or without administrator rights, and runs `check_export`, which must
   print two `PASS` lines.

## 5. Start the app

```bash
python -m uvicorn app.main:app --port 8026
```

Open <http://127.0.0.1:8026>. The screener portal is at `/screen`, and the
doctor's review queue is at `/portal/doctor`.

The app uses PyTorch when its model files are present, and MATLAB otherwise.
To force one engine, set `DRISHTI_BACKEND` before starting:

| Shell | Command |
|---|---|
| PowerShell | `$env:DRISHTI_BACKEND = "matlab"` |
| Command Prompt | `set DRISHTI_BACKEND=matlab` |
| bash / zsh | `export DRISHTI_BACKEND=matlab` |

The header of the screener portal shows which engine is running.

**With MATLAB, the first photograph after starting the app takes about two
minutes**, while MATLAB starts and loads its networks. Screen one sample before
a demonstration; later photos take seconds.

## 6. How fast is it?

Measured on the development laptop (RTX 4060 laptop GPU), with the GPU
switched off for the CPU column. Both columns passed the full browser test.

| Engine | With a GPU | CPU only |
|---|---|---|
| PyTorch | 1 to 2 s a photo | about 8 s a photo |
| MATLAB | 2 to 4 s a photo | about 10 s a photo |
| MATLAB, first photo after starting | about 1.5 min | about 2 min |

A CPU-only machine is fine for testing and demonstrations. To try the CPU path
on a machine that has a GPU: `CUDA_VISIBLE_DEVICES=-1` for PyTorch and
`DRISHTI_MATLAB_CPU=1` for MATLAB.

## 7. Check that everything works

```bash
python -m pytest -q                      # unit tests
python scripts/e2e_ui.py                 # clicks through every page in a headless browser
python scripts/e2e_ui.py --backend matlab
```

The browser test starts its own copy of the app on a throwaway database, so
your cases are untouched. It needs the Playwright browser once:
`python -m playwright install chromium`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError` (timm, simpy, ...) when starting the app | run `python scripts/bootstrap.py`; make sure `python` is the same interpreter each time (`python -c "import sys; print(sys.executable)"`) |
| "release ... not found" during setup | the repository is private: run `gh auth login`, and check that your account was added to the repository |
| The screener shows "Model files not found" | run `python scripts/bootstrap.py --models-only` and restart the app |
| MATLAB: "missing Deep Learning Toolbox Converter for ONNX Model Format" | install it from **Home > Add-Ons** in MATLAB |
| MATLAB Engine: "Python version not supported" | R2026b supports Python 3.10 to 3.14; older MATLAB releases support fewer. Use a Python version your MATLAB lists |
| The first photograph hangs with MATLAB | normal for about two minutes; the server log shows MATLAB loading ONNX files |
| Port 8026 already in use | add `--port 8030` (any free port) |
| Out of memory on a small laptop | close other programs; PyTorch needs about 3 GB of RAM, MATLAB about 6 GB |
