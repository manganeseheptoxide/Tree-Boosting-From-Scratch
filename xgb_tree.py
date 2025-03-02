import pandas as pd
import numpy as np
from loss_functions import *

def optimal_output(loss_fcn: LossFunction, regularization_param, actual, predicted):
    y_actual = np.array(actual)
    y_predicted = np.array(predicted)
    numerator = -np.sum(loss_fcn.gradient(y_actual, y_predicted))
    denominator = np.sum(loss_fcn.hessian(y_actual, y_predicted)) + regularization_param
    return numerator / denominator

def similarity_score(loss_fcn: LossFunction, regularization_param, actual, predicted):
    y_actual = np.array(actual)
    y_predicted = np.array(predicted)
    numerator = np.sum(loss_fcn.gradient(y_actual, y_predicted))**2
    denominator = np.sum(loss_fcn.hessian(y_actual, y_predicted)) + regularization_param
    return numerator / denominator

class XGB_Node:
    def __init__(self, X: pd.DataFrame, y: pd.Series, fcn_estimate,
                 loss_fcn: LossFunction, regularization_param: float,
                 is_Terminal: bool = True, split: float = None, feature: str = None,
                 lt_node=None, gte_node=None,  
                 parent=None, depth: int = 0,
                 generate: bool = False, max_depth: float = float('inf'), min_points: int = 1):
        self.X = X
        self.y = y
        self.fcn_estimate = fcn_estimate
        self.regularization_param = regularization_param
        self.loss_fcn = loss_fcn
        self.output = None
        self.similarity_score = None
        self.is_Terminal = is_Terminal
        self.split = split # Candidate split point
        self.feature = feature # Feature used for split
        self.lt_node = lt_node
        self.gte_node = gte_node
        self.parent = parent
        self.depth = depth
        self.generate = generate
        self.max_depth = max_depth
        self.min_points = min_points



    def set_estimate_fcns(self, fcn_estimate):
        self.fcn_estimate = fcn_estimate
        pred = self.fcn_estimate(self.X)
        self.output = optimal_output(self.loss_fcn, self.regularization_param, self.y, pred)
        self.similarity_score = similarity_score(self.loss_fcn, self.regularization_param, self.y, pred)

    def add_Nodes(self, lt_node, gte_node):
        self.lt_node = lt_node
        self.gte_node = gte_node
        self.lt_node.parent = self
        self.gte_node.parent = self
        self.is_Terminal = False

    def evaluate_point(self, features):
        if self.output is None:
            print("Set the output first!")
            return None
        if self.is_Terminal:
            return self.output
        else:
            if features[self.feature] < self.split:
                return self.lt_node.evaluate_point(features)
            else:
                return self.gte_node.evaluate_point(features)

    def create_child(self, generate: bool = False, max_depth: float = float('inf'), min_points: int = 1):
        def feature_best_split(X, y, X_all):
            best_ss = -float('inf')
            best_split = None
            for i in range(1, len(X)):
                s = (X.iloc[i-1] + X.iloc[i]) / 2  # Candidate split point
                left_mask = X <= s
                right_mask = X > s

                y_left, y_right = y[left_mask], y[right_mask]
                X_left, X_right = X_all[left_mask], X_all[right_mask]

                if len(y_left) == 0 or len(y_right) == 0:
                    continue  # Skip empty splits

                ss = (similarity_score(self.loss_fcn, self.regularization_param,
                                        y_left, self.fcn_estimate(X_left)) +
                      similarity_score(self.loss_fcn, self.regularization_param,
                                        y_right, self.fcn_estimate(X_right)))

                if ss > best_ss:
                    best_ss, best_split = ss, s
            return best_split, best_ss

        def best_split(X: pd.DataFrame, y: pd.Series):
            best_ss = -float('inf')
            best_s = None
            best_feature = None
            for feature in X.columns:
                X_sorted = X.sort_values(by=feature)
                split, ss = feature_best_split(X_sorted[feature], y[X_sorted.index], X_all=X_sorted)
                if ss > best_ss:
                    best_ss, best_s, best_feature = ss, split, feature
            return best_ss, best_s, best_feature

        _, best_s, best_feature = best_split(self.X, self.y)

        if best_s is not None and best_feature is not None:
            self.split = best_s
            self.feature = best_feature

            lt_mask = self.X[best_feature] < best_s
            gte_mask = self.X[best_feature] >= best_s

            lt_X, lt_y = self.X[lt_mask], self.y[lt_mask]
            gte_X, gte_y = self.X[gte_mask], self.y[gte_mask]

            if len(lt_X) > self.min_points and len(gte_X) > self.min_points:
                child_ss = (similarity_score(self.loss_fcn, self.regularization_param,
                                              lt_y, self.fcn_estimate(lt_X)) +
                            similarity_score(self.loss_fcn, self.regularization_param,
                                              gte_y, self.fcn_estimate(gte_X)))
                if child_ss > self.similarity_score + self.regularization_param:
                    lt_node = XGB_Node(X=lt_X, y=lt_y, fcn_estimate=self.fcn_estimate,
                                        loss_fcn=self.loss_fcn, regularization_param=self.regularization_param,
                                        depth=self.depth + 1, generate=generate, max_depth=max_depth, min_points=min_points)
                    gte_node = XGB_Node(X=gte_X, y=gte_y, fcn_estimate=self.fcn_estimate, 
                                        loss_fcn=self.loss_fcn, regularization_param=self.regularization_param,
                                        depth=self.depth + 1, generate=generate, max_depth=max_depth, min_points=min_points)
                    lt_node.set_estimate_fcns(self.fcn_estimate)
                    gte_node.set_estimate_fcns(self.fcn_estimate)
                    self.add_Nodes(lt_node=lt_node, gte_node=gte_node)

    def generate_children(self, max_depth: float, min_points: int):
        if self.is_Terminal:
            self.create_child(generate=True, max_depth=max_depth, min_points=min_points)

class XGB_Tree:
    def __init__(self, fcn_estimate=None,
                 loss_fcn: LossFunction = SSR, regularization_param: float = 0.0,
                 root_node: XGB_Node = None, X: pd.DataFrame = None, y: pd.Series = None):
        self.root_node = root_node       
        self.X = X
        self.y = y
        self.fcn_estimate = fcn_estimate
        self.regularization_param = regularization_param
        self.loss_fcn = loss_fcn
        self.features = None
        self.depth = None
        if self.y is not None:
            self.get_features()

    def load_data(self, X: pd.DataFrame, y: pd.Series):
        if self.X is None:
            self.X = X
            self.y = y
        else:
            print("Data has already been loaded!")

    def create_root_node(self):
        if self.root_node is None:
            if self.X is None:
                print("Load the data first!")
            else:
                self.root_node = XGB_Node(X=self.X, y=self.y, fcn_estimate=self.fcn_estimate,
                                          loss_fcn=self.loss_fcn, regularization_param=self.regularization_param)
                self.root_node.set_estimate_fcns(self.fcn_estimate)
        else:
            print("The tree already has a root node!")

    def evaluate(self, feature_set: pd.DataFrame):
        if self.root_node is None:
            print("Generate a tree first!")
        else:
            predictions = []
            for _, features in feature_set.iterrows():
                predictions.append(self.root_node.evaluate_point(features))
            return np.array(predictions)
        
    def get_features(self):
        self.features = self.X.columns.tolist()

    def generate_tree(self, max_depth: float = float('inf'), min_points: int = 1):

        if self.root_node is None:
            self.create_root_node()
        
        # Generate nodes breadth-first
        queue = [self.root_node]
        while queue:
            current_node = queue.pop(0)
            # Only generate children if depth and data size conditions hold.
            if current_node.depth < max_depth and current_node.X.shape[0] >= min_points:
                current_node.create_child(generate=True, max_depth=max_depth, min_points=min_points)
                if not current_node.is_Terminal:
                    # Append children nodes to the queue.
                    queue.append(current_node.lt_node)
                    queue.append(current_node.gte_node)
