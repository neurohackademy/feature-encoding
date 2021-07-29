Y (brain) = X (features)
Dimensionality reduction X: PCA  >> later, we would study the annotations and label the PCs into meaningful subcategories. 
Run separate lasso PCRs per dimension per subject (beta map x participant x feature)

File descriptions:
	resampledat.py -> downsample voxel size
	linear_model.py -> run linear model (LASSO) and store coefficient
	stat_t_test.py -> conduct group t test on coefficients
	plot_brain.py -> plot the group t statistics results
