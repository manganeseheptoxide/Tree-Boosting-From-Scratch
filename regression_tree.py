import pandas as pd
import numpy as np

class Node:
    def __init__(self, X: pd.DataFrame, y: pd.Series,
                 is_Terminal: bool = True, split: float = None, feature: str = None,
                 lt_node=None, gte_node=None,  
                 parent=None, depth: int = 0,
                 generate: bool = False, max_depth: float = float('inf'), min_points: int = 1):
        self.X = X
        self.y = y
        self.mean = y.mean()
        self.err = np.sum((y - self.mean)**2)
        self.is_Terminal = is_Terminal
        self.split = split
        self.feature = feature
        self.lt_node = lt_node
        self.gte_node = gte_node
        self.parent = parent
        self.depth = depth
        self.generate = generate
        self.max_depth = max_depth
        self.min_points = min_points
        # This allows nodes to recursively generate children nodes until the max_depth or the min_points has been reached.
        if self.generate and self.depth <= self.max_depth and self.X.shape[0] >= self.min_points:
            self.create_child(generate=self.generate, max_depth=self.max_depth, min_points=self.min_points)

    def add_Nodes(self, lt_node, gte_node):
        self.lt_node = lt_node
        self.gte_node = gte_node
        self.lt_node.parent = self
        self.gte_node.parent = self
        self.is_Terminal = False

    def evaluate_point(self, features):
        if self.is_Terminal:
            return self.mean
        else:
            if features[self.feature] < self.split:
                return self.lt_node.evaluate_point(features)
            else:
                return self.gte_node.evaluate_point(features)

    def create_child(self, generate: bool = False, max_depth: float = float('inf'), min_points: int = 1):
        def feature_best_split(X, y):
            X_sorted, y_sorted = np.array(X), np.array(y)

            best_sse = float('inf')
            best_split = None
            for i in range(1, len(X_sorted)):
                s = (X_sorted[i-1] + X_sorted[i]) / 2  # Candidate split point
                left_mask = X_sorted <= s
                right_mask = X_sorted > s

                y_left, y_right = y_sorted[left_mask], y_sorted[right_mask]

                if len(y_left) == 0 or len(y_right) == 0:
                    continue  # Skip empty splits

                sse = np.sum((y_left - y_left.mean())**2) + np.sum((y_right - y_right.mean())**2)

                if sse < best_sse: # Getting the best split that minimizes the SSE for that particular feature
                    best_sse, best_split = sse, s
            return best_split, best_sse

        def best_split(X: pd.DataFrame, y: pd.Series):
            best_sse = float('inf')
            best_s = None
            best_feature = None
            for feature in X.columns:
                X_sorted = X.sort_values(by=feature)
                # print(len(X_sorted))
                # print(len(y))
                split, sse = feature_best_split(X_sorted[feature], y[X_sorted.index])
                
                if sse < best_sse: # Getting the best split among all features that minimizes the SSE
                    best_sse, best_s, best_feature = sse, split, feature
            return best_sse, best_s, best_feature

        _, best_s, best_feature = best_split(self.X, self.y)

        if best_s is not None and best_feature is not None:
            self.split = best_s
            self.feature = best_feature

            lt_mask = self.X[best_feature] < best_s
            gte_mask = self.X[best_feature] >= best_s

            lt_X, lt_y = self.X[lt_mask], self.y[lt_mask]
            gte_X, gte_y = self.X[gte_mask], self.y[gte_mask]

            if len(lt_X) > self.min_points and len(gte_X) > self.min_points:
                lt_node = Node(X=lt_X, y=lt_y, depth=self.depth + 1,
                               generate=generate, max_depth=max_depth, min_points=min_points)
                gte_node = Node(X=gte_X, y=gte_y, depth=self.depth + 1,
                                generate=generate, max_depth=max_depth, min_points=min_points)
                self.add_Nodes(lt_node=lt_node, gte_node=gte_node)

    def generate_children(self, max_depth: float, min_points: int):
        if self.is_Terminal:
            # print("Cannot generate children from a terminal node!")
            self.create_child(generate=True, max_depth=max_depth, min_points=min_points)

class Tree:
    def __init__(self, root_node: Node = None, X: pd.DataFrame = None, y: pd.Series = None):
        self.root_node = root_node       
        self.X = X
        self.y = y
        self.features = None
        self.depth = None
        self.err = float('inf') # The training SSE of the tree
        self.residuals = None
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
                self.root_node = Node(X=self.X, y=self.y)
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

    def calculate_error(self):
        self.err = np.sum((self.y - self.evaluate(self.X))**2)

    def calculate_residuals(self):
        self.residuals = self.y - self.evaluate(self.X)

    def generate_tree(self, max_depth: float = float('inf'), min_points: int = 1):
        if self.root_node is None:
            self.create_root_node()
        self.root_node.generate_children(max_depth=max_depth, min_points=min_points)
        self.calculate_error()
        self.calculate_residuals()

