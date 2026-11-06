# EEG background noise analysis

My CS 163 contribution to [ABR Deconvolution](https://github.com/jduantc/ABR-Deconv). This notebook contains the noise analysis used in Figures 3–5 of our midterm report.

## Data and setup

Data: Polonenko and Eisenreich (2026), [OpenNeuro ds008065](https://openneuro.org/datasets/ds008065). The EEG is stored in BrainVision format (`.vhdr`, `.eeg`, `.vmrk`) with BIDS `_events.tsv` files. The dataset is **not** part of this repository.

Install dependencies in a Python/Jupyter environment:

```bash
pip install -r requirements.txt
```

Open `eeg_noise_analysis.ipynb` in Jupyter or VS Code. In the first code cell, set `DATA_ROOT` to the directory containing the `sub-exp...` folders (or keep `None` if the data are found automatically). Run all cells and save the notebook.

`MODE = "five"` reproduces the five Experiment 2 recordings (`sub-exp221`–`sub-exp225`) analyzed in the report. `MODE = "all"` runs the same code for every locally available recording. The expanded analysis can combine different experiments, so the experiment column should be kept when interpreting results.

## Analysis

- **Figure 3:** Pooled 1-ms adjacent-sample covariance/variance statistic, before and after filtering. This is an autocorrelation estimate under approximate stationarity; it is not exactly the finite-sample Pearson coefficient.
- **Figure 4:** Normalized sample covariance matrices using held-aside baseline windows, for the first 75 ms.
- **Figure 5:** Histogram and normal Q–Q plot for 20,000 randomly sampled unfiltered EEG values from participant 221, at 10 kHz.

For each stimulus onset, the notebook extracts EEG from −450 to −50 ms, using the `FCzA1` channel. Figures 3 and 4 use 1-kHz data; the initial bandpass is 30–2000 Hz before antialiased downsampling. Figure 5 uses the original 10-kHz samples. Generated PNGs and CSVs are written to `noise_results/` (ignored by Git); the saved notebook itself can display the graphs on GitHub.

## Limitations

The pre-stimulus windows are a proxy for background noise, not necessarily pure noise. The source paper excluded `sub-exp223` from its primary analysis because of recording problems; it is included here as an exploratory sample. The additional `/100` in EEG amplitude scaling was inherited from the original notebooks and should be checked against the recording metadata before interpreting absolute voltages. The main correlation and distribution-shape results are invariant to a fixed positive amplitude scale.

The notebook characterizes noise; it does not establish whether a new deconvolution method improves ABR recovery.
