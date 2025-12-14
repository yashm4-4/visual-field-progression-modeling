# Chem 277B — Machine Learning Algorithms
## Machine Learning Applications for Visual Field Analysis: A Study of Vision Loss Progression
Team 1 — UC Berkeley, College of Chemistry

Contributors: David Houshangi, Lily Hirano, Kirk Ehmsen, Christian Fernandez, Yash Maheshwaran

# 1. Project Overview

Glaucoma is a chronic, progressive disease that damages the optic nerve and can lead to irreversible blindness if not detected early. Because early stages often have no symptoms, nearly half of affected individuals do not know they have the disease. Visual field (VF) testing is one of the primary tools for diagnosing glaucoma and tracking its progression, as it measures the patient’s functional vision over time.

This project explores how modern machine-learning methods, including unsupervised learning, Random Forests, CNNs, and LSTMs can be used to analyze visual field maps, detect patterns of glaucomatous loss, and predict future progression.

We used two major datasets:

1. GRAPE Dataset: 263 eyes, 1,115 longitudinal VF tests

2. UW Biomedical AI Dataset: 3,871 patients, 28,943 VF tests

Together, these datasets allow us to study glaucoma progression using real functional measurements rather than structural imaging alone.

# 2. Project Goals

**2.1 Unsupervised Learning**

Cluster visual fields into meaningful glaucoma subtypes based on the spatial pattern and rate of deterioration.

**2.2 Gradient Boost Regression**

Predict long-term progression (MS slope) from baseline, early-window, and MS acceleration features. Identify the strongest feature predictors of decline.

**2.3 Random Forest Regression**

Predict long-term progression (MS slope) from baseline VF features and identify the strongest physiological predictors of decline. We also evaluated a classification setting by binning slopes into stable, slow, and fast progression groups, which yielded more reliable performance than direct slope prediction.

**2.4 CNN-Based VF Prediction from Color Fundus Photographs**

Train a convolutional neural network to predict VF sensitivity points (dB) by learning spatial patterns of retinal optic nerve head (ONH) damage in color fundus photographs paired with 58 VF data points multi-regression targets.

**2.5 LSTM-Based Progression Modeling**

Use longitudinal VF sequences to model temporal dynamics and predict how damage evolves over time.

**2.6 Dataset Alignment**

Standardize map formats so models trained on UW data can generalize to GRAPE, enabling cross-dataset comparisons.

# 3. Repository Structure

project-root/

README.md  

requirements.txt  

.gitignore   

data/

pkl_data/

EDA/

unsupervised_model/

RandomForest_model/

gradient_boosting/

CNN/

LSTM/
           

# 4. Methods 

- Unsupervised Learning: PCA + KMeans + UMAP for clustering VF progression patterns

- Gradient Boosting: Predict slope of MS loss; extract feature importances

- Random Forest Regression: Predict slope of MS loss; extract feature importances

- CNN Model: Predict VF sensitivity values from color fundus photographs

- LSTM Models: Sequence-based prediction of future sensitivity loss


# 5. References

1. GRAPE Dataset
Huang et al. GRAPE: A multi-modal dataset of longitudinal follow-up visual fields and fundus images for glaucoma management.

2. UW Biomedical AI Dataset
Wang et al. A large-scale clinical visual field database for glaucoma analysis.

3. Centers for Disease Control and Prevention. (2024, May 15). Fast facts: Vision loss. U.S. Department of Health and Human Services. https://www.cdc.gov/vision-health/data-research/vision-loss-facts/index.html
   
4. Huang, X., Kong, X., Shen, Z. et al. GRAPE: A multi-modal dataset of longitudinal follow-up visual field and fundus images for glaucoma management. Sci Data 10, 520 (2023). https://doi.org/10.1038/s41597-023-02424-4
   
5. Giovanni Montesano, Andrew Chen, Randy Lu, Cecilia S. Lee, Aaron Y. Lee; UWHVF: A Real-World, Open Source Dataset of Perimetry Tests From the Humphrey Field Analyzer at the University of Washington. Trans. Vis. Sci. Tech. 2022;11(1):2. doi: https://doi.org/10.1167/tvst.11.1.1.

6. Boden, C., et al. (2004). Patterns of glaucomatous visual field progression: Defects expand in areas of prior loss. American Journal of Ophthalmology, 138(5), 802–808. https://doi.org/10.1016/j.ajo.2004.07.00 
