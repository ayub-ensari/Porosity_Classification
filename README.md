# Porosity defect classification

This project detects porosity defects from metal additive manufacturing by employing convolutional neural networks (CNN). Laser Powder Bed Fusion (LPBF) is a selective laser melting technique used to build complex 3D parts. The current monitoring system in LPBF is inadequate to produce safety-critical parts due to the lack of automated processing of collected data. 
We formulated two labelling strategies based on the computer-aided design (CAD) file, and X-ray computed tomography (XCT) scan data.
A novel CNN was trained from scratch and optimised by selecting the best values of an extensive range of hyper-parameters by employing a Hyperband tuner. The model's accuracy was 90\% when trained using CAD-assisted labelling and 97\% when using XCT-assisted labelling. The model successfully spotted pores as small as 0.2mm. Experiments revealed that balancing the data set improved the model's precision from 89\% to 97\% and recall from 85\% to 97\% compared to training on an imbalanced data set. We firmly believe that the proposed model would significantly reduce post-processing costs and provide a better base model network for transfer learning of future ML models aimed at LPBF micro-defect detection. For more details, please read our published work [here](https://link.springer.com/article/10.1007/s00170-022-08995-7).

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install required python libraries from the requirements.txt file.

```bash
pip install -r requirements.txt
```



## Notebooks Descriptions
In a nutshell, here is a brief description of what to expect in different Jupiter notebooks.
1. 1_Data_Collection.ipynb: Explains the image reading, cropping images, labelling images and storing the final datasets.
2. 2_Image_Augmentation.ipynb: Covers the various image data augmentation methods
3. 31_Image_Brightness_Edge_Detection_Binarisation.ipynb: Explains different image enhancement methods.
4. 3_Image_Enhancement_Examples.ipynb: Explains image histogram analysis, thresholding and Otsu binarization.
5. 4_Model_Training_CAD.ipynb: Trains a CNN using CAD labelling technique.
6. 5_Model_Training_XCT.ipynb: Trains a CNN using the XCT labelling approach.
7. 6_Model_Training_XCT-Debugging.ipynb: Explore the model's performance in predicting various sizes of pores.
8. 7_1_Hyperparameter_Tuning - Part1.ipynb: Prepares data for hyper-parameter tuning.
9. 7_2_Hyper_Parameter_Tuning - Part2.ipynb: Test CNN model with a range of hyper-parameters using hyperband tuner.
10. 7_3_Hyper_Parameter_Tuning - Part3.ipynb: Visualise and analyse hyper-parameter results.
11. 8_Model_training_XCT_Balanced_Data.ipynb: Final model trained on XCT labelled images and using balanced data after upsampling minority class via data augmentation. 

