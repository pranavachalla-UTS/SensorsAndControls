# UTS 41014 — Sensors and Control integrated project

**Track B: vision-guided UR3 moving-target pick in MuJoCo**

Start with the **[project page](project/README.md)** for the model, calibration, observers, control design, measured validation status and reproduction commands.

| Folder | Purpose |
| --- | --- |
| `project/` | Submission code, project page, tests and reproducible evidence |
| `Robots/` | Original UR3 assets and licence |
| `docs/` | Assessment guide and submission editing notes |

From a graphics-enabled local Terminal:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r project/requirements.txt
python project/run_all.py --quick --no-video
```

Edit your YouTube link, actual team contributions and AI declaration in the labelled sections at the bottom of [project/README.md](project/README.md). The AI declaration is intentionally blank for now. [Editing guide](docs/SUBMISSION_EDITING.md).
