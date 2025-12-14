import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from umap import UMAP
import matplotlib.pyplot as plt
from sklearn.metrics import calinski_harabasz_score, silhouette_score
# from scipy.interpolate import Rbf

class Unsupervised_Model():
    def __init__(self, data):
        self.data = data

    @staticmethod
    def create_VFdata(df): 
        patients = []
        masks = []
        for i, row in df.iterrows():
            visits = []
            mask = []

            for val in row:
                if isinstance(val, float) and pd.isna(val):
                    visits.append(np.full(len(df.iloc[0,0]), np.nan, dtype=float))
                    mask.append(0)
                else:
                    visits.append(np.array(val, dtype=float))
                    mask.append(1)

            patients.append(np.stack(visits))
            masks.append(np.array(mask, dtype=int))
        return np.array(patients, dtype=float), np.array(masks, dtype=int)
    
    @staticmethod
    def scale_VF_data(patients, masks):
        flat = patients.reshape(-1, patients.shape[-1])
        mask_flat = masks.reshape(-1, 1)


        valid_rows = flat[mask_flat[:,0]==1]
        scaler = StandardScaler()
        scaler.fit(valid_rows) #only fit on rows with values

        flat_scaled = scaler.transform(flat)  #NaN ignored, still flat

        return flat_scaled, valid_rows

    @staticmethod
    def find_PCA_emb(patients, flat_scaled, valid_rows, n=7):
        N, T, P = patients.shape
        pca = PCA(n_components=n, random_state=30)
        pca.fit(valid_rows)
        flat_visits_emb = pca.transform(np.nan_to_num(flat_scaled, nan=0.0)) #make NaN 0.0 to avoid errors

        visit_emb = flat_visits_emb.reshape(N, T, n)

        return visit_emb

    @staticmethod
    def get_features(visit_emb, masks, n):
        '''
        Current features:
        mean slope - mean slope of vision loss progression per dB point
        mean embeddings - mean of the reduced vector dimensions
        '''
        diffs = np.diff(visit_emb, axis=1)
        mean_slope = np.sum(diffs * masks[:,1:,None], axis=1) / np.sum(masks[:,1:], axis=1)[:,None]

        mean_emb = np.sum(visit_emb * masks[:, :, None], axis=1) / np.sum(masks, axis=1)[:, None]

        patient_features = np.concatenate([mean_emb, mean_slope], axis=1)

        patient_features[:, n:] = StandardScaler().fit_transform(patient_features[:, n:]) #scale mean slope
        return patient_features

    @staticmethod
    def kmean_clustering(patient_features, c=4):
        kmeans = KMeans(n_clusters=c, random_state=30)
        labels = kmeans.fit_predict(patient_features)

        stage_map = {0: "Early", 1: "Severe", 2: "Moderate", 3: "Advanced"}
        labels = np.array([stage_map[c] for c in labels])

        return labels

    @staticmethod
    def _visualize(patient_features, labels):
        '''
        Helper function for make_visual
        
        patient_features: Calculated features array
        labels: Labels from KMeans clustering
        '''
        umap = UMAP(n_neighbors=9, min_dist=0.1)
        emb_2d = umap.fit_transform(patient_features)

        colors = {"Early": "#440154","Moderate": "#30678d","Advanced": "#35b779","Severe": "#fde725"}

        for stage in np.unique(labels):
            plt.scatter(emb_2d[labels == stage, 0], emb_2d[labels == stage, 1], color=colors[stage], s=50, label=stage)
 
        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
        plt.legend()
        plt.title('Glaucomatous VF Clustering')
        plt.savefig('unsupervised_model/pt_clustering.png')
        plt.close()

    def train(self, n=7, c=4):
        '''
        n - number of components for PCA
            (default 7)
        c - number of clusters for KMeans
            (default 4)
            WARNING: changing c will break the labeling system
        '''
        self.patients, self.masks = self.create_VFdata(self.data)
        flat_scaled, valid_rows = self.scale_VF_data(self.patients, self.masks)
        visit_emb = self.find_PCA_emb(self.patients, flat_scaled, valid_rows, n)
        self.patient_features = self.get_features(visit_emb, self.masks, n)
        self.labels = self.kmean_clustering(self.patient_features, c)
        
    def make_visual(self):
        '''
        Uses UMAP to Plot Patient VF Trajectory Embeddings.
        Visual saved as pt_clustering.png
        - Umap parameters : n_neighbors=8, min_dist=0.1
        '''
        self._visualize(self.patient_features, self.labels)

    def metrics(self):
        '''
        Prints Silhouette score and CH index
        '''
        score = silhouette_score(self.patient_features, self.labels)
        print("Silhouette Score:", score)
        score2 = calinski_harabasz_score(self.patient_features, self.labels)
        print("CH Index:", score2)



