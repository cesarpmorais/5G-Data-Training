import numpy as np

from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation, MeanShift, SpectralClustering, OPTICS, Birch
from sklearn.neural_network import BernoulliRBM

class Method:
    def __init__(self, name, function, params):
        self.name = name
        self.f = function
        self.params = params

def get_methods() -> list:
    return [
        Method('K-Means', KMeans, 
            {
                'n_clusters': range(2,15),
            }
        ),
        Method('DBSCAN', DBSCAN, 
            {
                'eps': [0.3, 0.5],
                'min_samples': [3, 5, 20, 50, 100]
            }
        ),
        Method('OPTICS', OPTICS,
            {
                'min_samples': [3, 5, 10],
                'max_eps': [0.5, 1.0]
            }
        ),
        Method('BIRCH', Birch,
            {
                'threshold': [0.3, 0.5, 0.7],
                'n_clusters': [2]
            }
        ),
        Method('MeanShift', MeanShift,
            {
                'bandwidth': [0.8, 1.0, 1.2]
            }
        ),
        Method('AffinityPropagation', AffinityPropagation,
            {
                'damping': [0.5, 0.75, 0.9],
                'max_iter': [100, 200, 400]
            }
        ),
        Method('SpectralClustering', SpectralClustering,
            {
                'n_clusters': [2],
                'affinity': ['nearest_neighbors', 'rbf']
            }
        ),
    ]