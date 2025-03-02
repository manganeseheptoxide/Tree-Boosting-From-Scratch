import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from loss_functions import *
from xgb_tree import *

class XGBoost:
    def __init__(self, loss_function: LossFunction = SSR, regularization_param: float = 0.0,
                 n_estimators: int = 100,
                 learning_rate: float = 0.01, max_depth: int = 1, min_points: int = 20):
        self.loss_function = loss_function
        self.regularization_param = regularization_param
        self.ensemble = []
        self.loss = []
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_points = min_points
        self.init_tree = None
        self.init_guess = None
        self.init_guess_mean = None
        self.last_guess = None
    
    def predict(self, X: pd.DataFrame, training: bool = False):
        predictions = np.zeros(X.shape[0])
        if training:
            predictions = self.last_guess
            
            predictions += self.learning_rate * self.ensemble[-1].evaluate(X)
            self.last_guess = predictions
        else:
            predictions = self.init_tree.evaluate(X) if self.init_tree is not None else np.full(X.shape[0], self.init_guess_mean)
            for tree in self.ensemble[1:]:
                predictions += self.learning_rate * tree.evaluate(X)

        return pd.Series(predictions, index=X.index)
    
    def predict_subset(self, X: pd.DataFrame):
        subset_indices = X.index
        predictions_subset = self.last_guess[subset_indices]
        return predictions_subset

    def fit(self, X: pd.DataFrame, y: pd.Series, init_tree: XGB_Tree = None, early_stopping: int = 5):
        self.init_tree = init_tree
        self.init_guess_mean = y.mean()
        self.init_guess = pd.Series(init_tree.evaluate(X), index=y.index) if init_tree is not None else pd.Series(np.full(y.shape, y.mean()), index=y.index)
        self.last_guess = self.init_guess.copy()
        if init_tree is not None:
            self.ensemble.append(init_tree)
        loss = self.loss_function.loss(y, self.init_guess)
        print(f"Iteration: 0, Loss: {loss:.4f}")
        loss_increase = 0
        self.loss.append(loss)
        for i in range(self.n_estimators):
            tree = XGB_Tree(X=X, y=y, fcn_estimate=self.predict_subset, loss_fcn=self.loss_function,
                            regularization_param = self.regularization_param)
            tree.generate_tree(max_depth=self.max_depth, min_points=self.min_points)
            self.ensemble.append(tree)
            predictions = self.predict(X, training=True)
            loss = self.loss_function.loss(y, predictions)
            if loss >= self.loss[-1]:
                loss_increase += 1
            self.loss.append(loss)
            print(f"Iteration: {i + 1}, Loss: {loss:.4f}")
            if loss_increase >= early_stopping:
                print(f"Early stopping at iteration {i + 1}")
                break
        return self
    
    def plot_loss(self):
        plt.plot(self.loss)
        plt.title("Loss vs. Iterations")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.show()