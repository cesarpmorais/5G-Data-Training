import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

class Method:
    def __init__(self, name, function, params):
        self.name = name
        self.f = function
        self.params = params

def get_methods() -> list:
    return [
        Method('k-neighbors', KNeighborsClassifier, 
            { 
                'n_neighbors': np.arange(1,10),
                'weights': ['uniform', 'distance'],
                'leaf_size': [20, 30, 50, 100],
            }
        ),

        Method('supervised_neural_network', MLPClassifier, 
            {
                'hidden_layer_sizes': [(50,), (100,), (150,), (100, 50), (100, 100, 50)],
                'learning_rate_init': [0.0001, 0.001, 0.01, 0.1],
                'alpha': [0.0001, 0.001, 0.01, 0.1]
            }
        ),

        Method('naive_bayes', GaussianNB,
            {
                'priors': [None, [0.5, 0.5]],
                'var_smoothing': [1e-9, 1e-7, 1e-5],
            }
        ),

        Method('ada_boost', AdaBoostClassifier, 
            {
                'estimator': [None, DecisionTreeClassifier(max_depth=2), DecisionTreeClassifier(max_depth=3)],
                'n_estimators': [25, 50, 100],
                'learning_rate': [0.5, 1.0, 2.0],
                'algorithm': ['SAMME']
            }
        ),
        
        Method('random_forest', RandomForestClassifier, {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [None, 10, 20, 30, 40, 50],
            'max_features': ['sqrt', 'log2', None]
        }),

        Method('gradient_boosted_trees', HistGradientBoostingClassifier, {
            'learning_rate': [0.01, 0.1, 0.2],
            'max_iter': [100, 300],
            'max_leaf_nodes': [10, 31, 50]
        }),

    ]