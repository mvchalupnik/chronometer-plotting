# Chronometer Plotting Classes

This repo contains classes which allow a user to load CSVs from Chronometer and other health tracking-related interfaces, process, and plot this data.

See example script (example.py) for typical usage.

Requires `scikit-learn`, `matplotlib`, `pandas`, and `seaborn`.

## Example output plots

Some examples of output plots:

Heatmap displaying Pearson correlation:

<img src="imgs/heatmap-gluco_keton_Energ_imputer-None.png" width = "600">

Linear regression with scatter plot:

<img src="imgs/linearreg-gluco_keton_Energ_imputer-None.png" width = "600">

Energy plotted against day:

<img src="imgs/myfile_CHRONOMETER_Energy_resampleby-DAY_nrows-None_startat-2023-01-01.png" width = "600">

Glucose reading plotted against day:

<img src="imgs/myfile_KETOMOJO_glucose_resampleby-DAY_nrows-None_startat-2023-01-01.png" width = "600">
