# TCDGroup-HPC-Project

Dataset: https://github.com/EdgarLopezPhD/PaySim

# Project Description

Using dataset 'Paysim' to build fraud detection models to preditct fraud rate. Using Back Propagation model/Random Forest model/Logestic Regression model as serial algorithms，then using sequential tensorflow as parallel algorithm to parallel models.

- paysim_data_description.ipynb: The dataset description as jupyter note shows. 6M transections be built as file 'PS_20174392719_1491204439457_log'.
- preprocess.py: read and clean the dataset.
- build_parameters.py: Building Parameter Space for BP/RF/LR models as bpparams/rfparams/lrparaps.
- build_models.py: Building BP/RF/LR models as BPmodel/RFmodel/LRmodel.
- normal.py: serial algorithms for BP/RF/LR models. Running seperately for  BPmodel(line 21-29)/RFmodel(line 31-39)/LRmodel(line 41-49)
- multi.py: parallel algorithm for BP/RF/LR models. Running seperately for  BPmodel(line 32-43)/RFmodel(line 46-59)/LRmodel(line 62-75)

#Result analysis

I only using 20 groups of parameter and 2,000 data samples to test BP/RF/LR models in my own laptop for testing time. I also use full dataset and full parameters to test BP model only. The results shows as below:

- Time for BP/RF/LR models' serial/parallel algorithms (with 20 groups of parameter and 2,000 data samples):
![0CADFDDE-1D62-40F8-AD2E-8EA30CAF234A](https://user-images.githubusercontent.com/39356710/125264393-68beb580-e336-11eb-9da9-899e2d635660.png)

- Time for BP model's serial/parallel algorithms (with 768 groups of parameter and 6,000,000 data samples):
![D83039E9-74AB-47F7-9E74-DFB056FAD40F](https://user-images.githubusercontent.com/39356710/125265013-fef2db80-e336-11eb-947e-1d5b28522c65.png)
