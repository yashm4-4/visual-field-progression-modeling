import pandas as pd
from unsupervised_model import Unsupervised_Model
from analyze_uns_model import Analyzer
import numpy as np


octopus_coords = np.array([
    (0,0), (1,-1), (-1,-1), (-1,1), (1,1), (2,-2), (-2,-2), (-2,2),
    (2,2), (4,-1), (4,-4), (1,-4), (-1,-4), (-4,-4), (-4,-1), (-4,1), (-4,4),
    (-1,4), (1,4), (4,4), (4,1), (6,0), (6,-5), (2,-6), (-2,-6), (-6,-5),
    (-6,-1), (-6,1), (-6,5), (-2,6), (2,6), (6,5), (8,-2), (8,-5), (8,-8), (5,-8), (2,-8), 
    (-2,-8), (-5,-8), (-8,-8), (-8,-5), (-8,-2), (-8,2), (-8,5), (-8,8), (-5,8), (-2,8), (2,8), (5,8),
    (8,8), (8,5), (8,2), (10,-3), (3,-10), (-3,-10), (-10, -2), (-10,2), (-3,10), (3,10), (10,3)
])

data = pd.read_pickle('./pkl_data/GRAPE.pkl')
grape_model = Unsupervised_Model(data)
grape_model.train(n=1) #n=1 most ideal for silhouette score
grape_model.make_visual()
grape_model.metrics()

analyzer = Analyzer(grape_model.labels, grape_model.patients, grape_model.masks, octopus_coords)

analyzer.run_all()

# np.save("./pkl_data/unsupervised_model_labels.npy", grape_model.labels)