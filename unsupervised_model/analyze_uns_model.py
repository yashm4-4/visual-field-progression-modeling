from scipy.interpolate import Rbf
import numpy as np
import matplotlib.pyplot as plt

class Analyzer():
    '''
    Analyzes the clusters of visual fields
    Saves images to the unsupervised_model folder
    '''
    def __init__(self, labels, patients, masks, coords):
        self.coords = coords
        self.patients = patients
        self.masks = masks
        self.labels = labels
        self.x_label, self.counts = np.unique(self.labels, return_counts=True)    #label of each cluster and num in each
        self.clusters = {}
        for c in np.unique(labels):
            self.clusters[c] = np.where(labels == c)[0]

        self.n_clusters = len(self.clusters)
        self.N, self.T, self.P = self.patients.shape

    def _plot_interpolated_vf(self, values, ax=None, cmap='viridis', title=None, vmin=None, vmax=None):
        """
        Plots one VF

        values: (60,) vector
        ax: matplotlib axis
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(6,6))

        grid_x, grid_y = np.mgrid[-12:12:200j, -12:12:200j] # interpolation grid

        rbf = Rbf(self.coords[:,0], self.coords[:,1], values, function='thin_plate')
        grid_z = rbf(grid_x, grid_y)

        radius = 11.5
        mask = np.sqrt(grid_x**2 + grid_y**2) <= radius     # circular mask
        masked_grid = np.where(mask, grid_z, np.nan)

        image = ax.imshow(masked_grid.T, extent=(-12, 12, -12, 12), origin='lower', cmap=cmap,
                        vmin = vmin, vmax = vmax) #plot

        ax.set_title(title if title else "")
        ax.axis('off')
        
        return image
    
    def _plot_interp_feature(self, feature, cmap = 'viridis', title = '', use_vminmax = 1):
        '''
        Plots a series of VFs.
        use_vminmax : choose from three options
            0 - create one colorbar based on vmin = 0 and vmax = 35 (Octopus dB range)
            1 - create one colorbar based on vmin and vmax of clusters
            2 - create many colorbars on individual scales
        '''
        fig, axes = plt.subplots(1, self.n_clusters, figsize=(4*self.n_clusters, 5))

        if use_vminmax == 0: #If want one colorbar on range 0 to 35
            for ax, c in zip(axes, self.clusters):
                values = feature[c]
                im = self._plot_interpolated_vf(values, ax=ax,
                                        cmap=cmap, title=c, vmin=0, vmax=35)
                
            fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)

        elif use_vminmax == 1: #If want one colorbar on same scale 
            all_vals = np.concatenate([feature[c] for c in self.clusters])
            vmin, vmax = all_vals.min(), all_vals.max()
            for ax, c in zip(axes, self.clusters):
                values = feature[c]
                im = self._plot_interpolated_vf(values, ax=ax,
                                        cmap=cmap, title=c, vmin=vmin, vmax=vmax)

            fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
            
        elif use_vminmax == 2: #If want many colorbars on individual scales
            for ax, c in zip(axes, self.clusters):
                values = feature[c]
                im = self._plot_interpolated_vf(values, ax=ax,
                                        cmap=cmap, title=c, vmin=None, vmax=None)
                fig.colorbar(im, ax=ax, shrink=0.6)

        else: raise ValueError(f"use_vminmax must be 0, 1, or 2. Got {use_vminmax}")
        
        fig.suptitle(title, fontsize=25)
        return plt

    def show_label_distribution(self):
        '''
        Bar plot of the label distributions
        '''
        plt.figure(figsize=(6,4))
        plt.bar(self.x_label, self.counts, color='grey')
        plt.xlabel("Cluster")
        plt.ylabel("Number of Patients")
        plt.title("Patient Distribution Across Clusters")
        plt.xticks(self.x_label)
        plt.savefig('unsupervised_model/label_distribution.png')

    def show_visit_distribution(self):
        '''
        Bar plot of average number of visits in each cluster
        '''
        avg_visits = []
        for c in self.x_label:
            i = self.clusters[c]     # compute average visits per cluster
            avg_visits.append(self.masks[i].sum(axis=1).mean())

        plt.figure(figsize=(6,4))
        plt.bar(self.x_label, avg_visits, color='lightblue')
        plt.xlabel("Cluster")
        plt.ylabel("Number of Patients")
        plt.title("Average Number of Visits per Patient by Cluster")
        plt.xticks(self.x_label)
        plt.savefig('unsupervised_model/visit_distribution.png')

    def show_mean_baseline(self):
        '''
        Interpolated VFs for the mean baseline in each cluster
        '''
        self.baseline_means = {c: np.nanmean(self.patients[idx, 0, :], axis=0)
                          for c, idx in self.clusters.items()}
        
        plot = self._plot_interp_feature(self.baseline_means, cmap='viridis', title = 'Mean Baseline VF per Cluster', use_vminmax=0)
        plot.savefig('unsupervised_model/mean_baseline.png')

    def _last_visit_per_patient(self):
        '''
        Finds the last visit for each patient
        '''
        last_visits = np.full((self.N, self.P), np.nan, dtype=float)
        for i in range(self.N):
            valid_idx = np.where(self.masks[i] == 1)[0]
            if valid_idx.size > 0:
                last_idx = valid_idx[-1]
                last_visits[i] = self.patients[i, last_idx]
        return last_visits

    def show_mean_final(self):
        '''
        Interpolated VFs for the mean final visit in each cluster
        '''
        last_visits = self._last_visit_per_patient()

        self.last_means = {c: np.nanmean(last_visits[idx], axis=0)
                           for c, idx in self.clusters.items()}
        
        plot = self._plot_interp_feature(self.last_means, cmap='viridis', title = 'Mean Final VF per Cluster', use_vminmax=0)
        plot.savefig('unsupervised_model/mean_final.png')

    def show_mean_change(self):
        '''
        Interpolated VFs for the mean change in each cluster
        On one color scale for direct cluster comparison
        '''
        self.diff_means = {c: self.last_means[c]- self.baseline_means[c] for c in self.clusters.keys()}
    
        plot = self._plot_interp_feature(self.diff_means, cmap='coolwarm_r', title = 'Mean Change in VF by Cluster', use_vminmax=1)
        plot.savefig('unsupervised_model/mean_change.png')

    def show_mean_change_per_cluster(self):
        '''
        Interpolated VFs for the mean change in each cluster
        On individual color scale
        '''
        plot = self._plot_interp_feature(self.diff_means, cmap='coolwarm_r', title = 'Mean Change in VF Within Each Cluster', use_vminmax=2)
        plot.savefig('unsupervised_model/mean_change_individual.png')

    def run_all(self):
        '''
        Creates all plots
        '''
        print('Label Distribution...')
        self.show_label_distribution()
        print('Visit Distribution...')
        self.show_visit_distribution()
        print('Mean Baseline...')
        self.show_mean_baseline()
        print('Mean Final...')
        self.show_mean_final()
        print('Mean Change...')
        self.show_mean_change()
        print('Mean Change Per Cluster...')
        self.show_mean_change_per_cluster()
            
        print('Done')
            




    


        


        
        
        

    

    
    