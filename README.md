This repository contains the materials required to replicate the analysis presented in:

Satta E. — Why the Euro Area Escaped Fiscal Inflation.

Files included

	•	ECB_API.ipynb – Jupyter notebook explaining how the Euro Area data are retrieved and transformed.
	
	•	ECB_functions.py – Python file containing all functions used in the notebook.
	
	•	model.mod – Dynare model file.
	
	•	launch.m – MATLAB script that launches the model estimation.

How to use

	1.	Run ECB_API.ipynb to download and preprocess the data.
	
	2.	Use ECB_functions.py for the functions required in the data pipeline.
	
	3.	Estimate the model by running Dynare on model.mod via the MATLAB script launch.m.

Make sure to adjust file paths in the scripts (ECB_API.ipynb, launch.m, and Dynare calls) to match your local directory structure before running the code.
