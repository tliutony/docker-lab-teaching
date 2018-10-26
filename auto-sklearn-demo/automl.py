"""
A basic demonstration of auto-sklearn.
(c) Tony Liu 2018.
"""

import sklearn
import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score
from autosklearn.classification import AutoSklearnClassifier

rand_seed = 2
np.random.seed(rand_seed)

# load MNIST
X, y = datasets.load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=rand_seed)
# this is a comment
# constrain automl so that it runs in reasonable time for demo
automl = AutoSklearnClassifier(
            time_left_for_this_task=20,
            per_run_time_limit=10,
            ml_memory_limit=1024,
            seed=rand_seed,
            ensemble_size=1, 
            initial_configurations_via_metalearning=0)
automl.fit(X_train, y_train)
y_hat = automl.predict(X_test)
print("Accuracy score", accuracy_score(y_test, y_hat))