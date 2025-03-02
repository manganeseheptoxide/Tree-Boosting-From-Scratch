from sklearn.tree import DecisionTreeRegressor, plot_tree
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def vis_regtree():
    X = np.random.rand(100, 1) * 10
    y = X.ravel() ** 2 + np.random.randn(100) * 10

    regressor = DecisionTreeRegressor(max_depth=3, random_state=0)
    regressor.fit(X, y)

    x_min, x_max = X.min(), X.max()
    x_grid = np.linspace(x_min, x_max, 500).reshape(-1, 1)
    y_pred = regressor.predict(x_grid)

    thresholds = np.sort(regressor.tree_.threshold[regressor.tree_.threshold > 0])

    colors = ListedColormap(["lightblue", "lightgreen", "lightpink", "lightyellow", "lavender"])

    plt.figure(figsize=(12, 8))
    plt.scatter(X, y, label="Data", color="blue", alpha=0.6)

    for i in range(len(thresholds) + 1):
        if i == 0:
            region_start = x_min
        else:
            region_start = thresholds[i - 1]
        
        if i == len(thresholds):
            region_end = x_max
        else:
            region_end = thresholds[i]
        
        plt.axvspan(region_start, region_end, color=colors(i % len(colors.colors)), alpha=0.4)

    for i, threshold in enumerate(thresholds):
        plt.axvline(x=threshold, color=f"C{i}", linestyle="--", linewidth=2, label=f"Boundary {i+1}")
    plt.plot(x_grid, y_pred, label="Mean", color="red", linewidth=2)
    plt.title("Decision Tree Regression with Colored Regions", fontsize=16)
    plt.xlabel("Feature (X)", fontsize=14)
    plt.ylabel("Target (y)", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.4)
    plt.show()

    plt.figure(figsize=(12, 8))
    plot_tree(regressor, filled=True, feature_names=["X"], fontsize=10)
    plt.show()
