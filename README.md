# Polarcam Live — README

This README is for the current **polarcam live** GUI in `polarcam_live/Spinners_gui_live.py`.

## Quick start (run the GUI)

```bash
# 1) create & activate venv
python -m venv .venv
. .venv/Scripts/Activate.ps1    # Windows PowerShell
# or: source .venv/bin/activate  # macOS/Linux
python -m pip install -U pip

# 2) install Python deps used by the GUI
python -m pip install -r polarcam_live/requirements.txt

# 3) launch (run from the repo root)
python polarcam_live/Spinners_gui_live.py
```

## What’s installed via `requirements.txt`

`polarcam_live/requirements.txt` installs the Python packages used for the GUI and analysis:

* `numpy`
* `opencv-python`
* `Pillow`
* `scipy`
* `matplotlib`

## What must be installed separately (not in `requirements.txt`)

These are needed **only for live camera capture / fetching frames**:

* **IDS peak SDK + drivers** (install IDS peak Cockpit / SDK from IDS).
* **IDS Python wheels** (after SDK install):
  * `ids-peak`
  * `ids-peak-ipl`
* **PySide6** (Qt runtime used by the IDS camera backend):
  * `python -m pip install PySide6`

If you only want to load and analyze saved AVI/NPY files in the GUI, you can skip the IDS SDK + `ids-peak*` installs.

## How to use the GUI

### Live video tab

* **Start live feed** starts the IDS camera stream.
* Adjust **exposure (ms)** and **gain**, then click **Apply**.
* Click the live image to update the magnifier ROI (if enabled).

### Spot analysis tab

* **Select AVI/NPY** to load a saved file.
* **Fetch frames** (optional) captures a short burst from the camera into an NPY stack.
* The S‑map overview shows detected spots; click a spot to update plots.
* Use **Update analysis** after changing DoG/phi/ring parameters.

### Spot examine tab

* Pick a spot ROI center and start a recording.
* Live plots show XY, phi(t), FFT, and handedness metrics.
* Save or discard recordings from this tab.

## Program summary

Polarcam Live is a Tkinter-based GUI that works with IDS polarization cameras and saved AVI/NPY files. It provides:

* Live preview with exposure/gain controls and magnifier view.
* Spot detection and an S‑map overview for navigation.
* Per‑spot analysis (XY scatter, phi(t), FFT) and rotation directionality metrics.
* Spot‑focused recording tools with quick preview and plotting.
