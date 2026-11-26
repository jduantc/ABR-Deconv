MLP vs U-Net reconstruction comparison

Status: complete.

This folder contains REPORT.html,
figures/01_old_data_rmse.png, figures/02_new_data_rmse.png,
figures/03_hybrid_example.png, figures/04_real_eeg_estimates.png,
SVG/PDF versions, guidance selection, protocol and per-recording data.

Old-data RMSE: existing held-out synthetic EEG experiment.
New-data RMSE: known synthetic ABRs with actual pulse times and recorded
quiet EEG; validation and test subjects are separated.
Actual speech EEG output is exploratory and cannot be assigned true RMSE
without a known clean ABR. It uses a single regressor even though other bands
also contribute to the recording.

The follow-up ran as the user service abr-eeg-followup.service.
