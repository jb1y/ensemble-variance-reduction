import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

data = load_breast_cancer()
data = pd.DataFrame(data.data, columns=data.feature_names)

data.head()

# Create table of malignant vs benign counts
data['target'] = load_breast_cancer().target
data['target'].value_counts()

train, test = train_test_split(
    data,
    test_size=0.4,
    random_state=100,
    stratify=data['target'],
)

x_train = train.drop(columns=['target'])
y_train = train['target']
x_test = test.drop(columns=['target'])
y_test = test['target']

x_train = np.asarray(x_train)
y_train = np.asarray(y_train).astype(int)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test).astype(int)

n_train = x_train.shape[0]
n_test = x_test.shape[0]

##########################################################################
# CROSS VALIDATION
##########################################################################

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=100,  # for reproducible folds
)

max_depth_values = [1, 2, 3, 4, 5, 6, 8, 10, None]

results_dt = []

for depth in max_depth_values:
    dt = DecisionTreeClassifier(
        max_depth=depth,
        random_state=100,
    )

    cv_res = cross_validate(
        dt,
        x_train,
        y_train,
        cv=cv,
        scoring='accuracy',
        return_train_score=True,
        n_jobs=-1,
    )

    results_dt.append({
        "max_depth": depth,
        "mean_train_acc": cv_res["train_score"].mean(),
        "mean_test_acc": cv_res["test_score"].mean(),
    })

results_dt = pd.DataFrame(results_dt)
print(results_dt)

k = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]

results_knn = []

for neighbors in k:
    knn = KNN(n_neighbors=neighbors)

    cv_res = cross_validate(
        knn,
        x_train,
        y_train,
        cv=cv,
        scoring='accuracy',
        return_train_score=True,
        n_jobs=-1,
    )
    results_knn.append({
        "n_neighbors": neighbors,
        "mean_train_acc": cv_res["train_score"].mean(),
        "mean_test_acc": cv_res["test_score"].mean(),
    })

results_knn = pd.DataFrame(results_knn)
print(results_knn)

###########################################################################
# KNN BAGGING
###########################################################################


def knn_ensemble(x_train, y_train, x_test, n_estimators=50, k=2, seed=100):
    rng = np.random.default_rng(seed)

    base_preds = np.zeros((n_test, n_estimators), dtype=int)

    for m in range(n_estimators):
        idx = rng.integers(0, n_train, size=n_train)
        X_boot = x_train[idx]
        y_boot = y_train[idx]

        model = make_pipeline(
            StandardScaler(),
            KNN(n_neighbors=k),
        )
        model.fit(X_boot, y_boot)

        base_preds[:, m] = model.predict(x_test)

    ensemble_pred = (base_preds.mean(axis=1) >= 0.5).astype(int)
    return ensemble_pred, base_preds


############################################################################
# TREE BAGGING
############################################################################


def tree_ensemble(x_train, y_train, x_test, n_estimators=50, max_depth=None, seed=100):
    rng = np.random.default_rng(seed)

    base_preds = np.zeros((n_test, n_estimators), dtype=int)

    for m in range(n_estimators):
        # Bootstrap sample within the training set
        idx = rng.integers(0, n_train, size=n_train)
        X_boot = x_train[idx]
        y_boot = y_train[idx]

        # Fit one tree on this bootstrap sample
        tree = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=seed + m,  # make tree construction deterministic
        )
        tree.fit(X_boot, y_boot)

        base_preds[:, m] = tree.predict(x_test)

    # Majority vote across trees
    ensemble_pred = (base_preds.mean(axis=1) >= 0.5).astype(int)
    return ensemble_pred, base_preds


###########################################################################
# LOGISTIC REGRESSION BAGGING
###########################################################################


def logistic_ensemble(x_train, y_train, x_test, n_estimators=50, seed=100):
    rng = np.random.default_rng(seed)

    n_train = x_train.shape[0]
    n_test = x_test.shape[0]

    base_preds = np.zeros((n_test, n_estimators), dtype=int)

    for m in range(n_estimators):
        idx = rng.integers(0, n_train, size=n_train)
        X_boot = x_train[idx]
        y_boot = y_train[idx]

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(random_state=seed + m, max_iter=5000),
        )

        model.fit(X_boot, y_boot)
        base_preds[:, m] = model.predict(x_test)

    ensemble_pred = (base_preds.mean(axis=1) >= 0.5).astype(int)
    return ensemble_pred, base_preds


###########################################################################
# BOOTSTRAP ESTIMATION FOR BAGGED TREES
###########################################################################


def boot_tree_bagged(x_train, y_train, x_test, y_test,
                     n_boot=200, seed=100, max_depth=None, n_estimators=50):
    rng_outer = np.random.default_rng(seed)

    # Each column = one bagged-tree ensemble trained on an outer bootstrap sample
    preds = np.zeros((n_test, n_boot), dtype=int)

    for b in range(n_boot):
        # Outer bootstrap sample of the training set
        idx_outer = rng_outer.integers(0, n_train, size=n_train)
        X_outer = x_train[idx_outer]
        y_outer = y_train[idx_outer]

        # Inner bagging: train an ensemble on X_outer/y_outer
        ensemble_pred, _ = tree_ensemble(
            X_outer, y_outer, x_test,
            n_estimators=n_estimators,
            max_depth=max_depth,
            seed=seed + b,
        )

        preds[:, b] = ensemble_pred

    # 0-1 loss decomposition (main prediction = majority vote over outer rounds)
    means = preds.mean(axis=1)
    main = (means >= 0.5).astype(int)

    bias_vec = np.where(main != y_test, 1, 0)
    disagreement = np.where(preds != main[:, None], 1, 0)
    var_vec = disagreement.mean(axis=1)

    return bias_vec.mean(), var_vec.mean(), preds


bias_bag_tree, var_bag_tree, preds_bag_tree = boot_tree_bagged(
    x_train, y_train, x_test, y_test,
    n_boot=500,
    seed=100,
    max_depth=8,
    n_estimators=5,
)

print(bias_bag_tree, var_bag_tree)

n_trees = [1, 5, 10, 20, 30, 50, 100, 200, 500]
indices = list(range(len(n_trees)))

trees = np.zeros((len(n_trees), 3))

for i in indices:
    bias_bag_tree, var_bag_tree, preds_bag_tree = boot_tree_bagged(
        x_train, y_train, x_test, y_test,
        n_boot=500,
        seed=100,
        max_depth=5,
        n_estimators=n_trees[i],
    )
    trees[i, 0] = n_trees[i]
    trees[i, 1] = bias_bag_tree
    trees[i, 2] = var_bag_tree

trees = pd.DataFrame(trees, columns=['n_estimators', 'bias', 'variance'])
print(trees)

############################################################################
# BOOTSTRAP ESTIMATION FOR BAGGED KNN
############################################################################


def boot_knn_bagged(x_train, y_train, x_test, y_test,
                    n_boot=200, seed=100, k=2, n_estimators=50):

    rng_outer = np.random.default_rng(seed)

    # Each column = one *bagged-KNN* model trained in one bootstrap round
    preds = np.zeros((n_test, n_boot), dtype=int)

    for b in range(n_boot):
        # Outer bootstrap = resample the training set
        idx_outer = rng_outer.integers(0, n_train, size=n_train)
        X_outer = x_train[idx_outer]
        y_outer = y_train[idx_outer]

        # Inner bagging = fit an ensemble on that resampled training set
        # Use a different seed per round so the inner bagging differs each time
        ensemble_pred, _ = knn_ensemble(
            X_outer, y_outer, x_test,
            n_estimators=n_estimators,
            k=k,
            seed=seed + b,
        )

        preds[:, b] = ensemble_pred

    # 0-1 loss decomposition (mode/majority-vote main prediction)
    means = preds.mean(axis=1)
    main = (means >= 0.5).astype(int)

    bias_vec = np.where(main != y_test, 1, 0)
    disagreement = np.where(preds != main[:, None], 1, 0)
    var_vec = disagreement.mean(axis=1)

    return bias_vec.mean(), var_vec.mean(), preds


n_ensemb = [1, 5, 10, 20, 30, 50, 100, 200, 500]
neighbours = np.zeros((len(n_ensemb), 3))

for i in indices:
    bias_bag_knn, var_bag_knn, preds_bag_knn = boot_knn_bagged(
        x_train, y_train, x_test, y_test,
        n_boot=500,
        seed=100,
        k=2,
        n_estimators=n_ensemb[i],
    )
    neighbours[i, 0] = n_ensemb[i]
    neighbours[i, 1] = bias_bag_knn
    neighbours[i, 2] = var_bag_knn

neighbours = pd.DataFrame(neighbours, columns=['n_estimators', 'bias', 'variance'])
print(neighbours)

############################################################################
# BOOTSTRAP ESTIMATION FOR BAGGED LOGISTIC REGRESSION
############################################################################


def boot_logistic_bagged(x_train, y_train, x_test, y_test,
                         n_boot=200, seed=100, n_estimators=50):

    rng_outer = np.random.default_rng(seed)

    # Each column = one bagged-logistic model trained in one bootstrap round
    preds = np.zeros((n_test, n_boot), dtype=int)

    for b in range(n_boot):
        # Outer bootstrap = resample the training set
        idx_outer = rng_outer.integers(0, n_train, size=n_train)
        X_outer = x_train[idx_outer]
        y_outer = y_train[idx_outer]

        # Inner bagging = fit an ensemble on that resampled training set
        # Use a different seed per round so the inner bagging differs each time
        ensemble_pred, _ = logistic_ensemble(
            X_outer, y_outer, x_test,
            n_estimators=n_estimators,
            seed=seed + b,
        )

        preds[:, b] = ensemble_pred

    # 0-1 loss decomposition (mode/majority-vote main prediction)
    means = preds.mean(axis=1)
    main = (means >= 0.5).astype(int)

    bias_vec = np.where(main != y_test, 1, 0)
    disagreement = np.where(preds != main[:, None], 1, 0)
    var_vec = disagreement.mean(axis=1)

    return bias_vec.mean(), var_vec.mean(), preds


log_reg = np.zeros((len(n_ensemb), 3))

for i in indices:
    bias_bag_log, var_bag_log, preds_bag_log = boot_logistic_bagged(
        x_train, y_train, x_test, y_test,
        n_boot=500,
        seed=100,
        n_estimators=n_ensemb[i],
    )
    log_reg[i, 0] = n_ensemb[i]
    log_reg[i, 1] = bias_bag_log
    log_reg[i, 2] = var_bag_log

log_reg = pd.DataFrame(log_reg, columns=['n_estimators', 'bias', 'variance'])
print(log_reg)

###########################################################################
# PLOTTING
###########################################################################

# Combine into one long dataframe (best for plotting)
knn_plot = neighbours.assign(model="KNN")
logit_plot = log_reg.assign(model="Logistic")
tree_plot = trees.assign(model="Tree")
df_all = pd.concat([knn_plot, logit_plot, tree_plot], ignore_index=True)

# ---- Plot 2: Variance ----
plt.figure()
for name, g in df_all.groupby("model"):
    plt.plot(g["n_estimators"], g["variance"], marker="o", label=name)

plt.xscale("log")
plt.xlabel("n_estimators (log scale)")
plt.ylabel("Variance")
plt.title("Variance vs Ensemble Size")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

plt.tight_layout()
plt.savefig("variance_vs_ensemble.pdf")        # vector (great for LaTeX)
plt.savefig("variance_vs_ensemble.png", dpi=300)
plt.show()

# ---- Plot 1: Bias ----
plt.figure()
for name, g in df_all.groupby("model"):
    plt.plot(g["n_estimators"], g["bias"], marker="o", label=name)

plt.xscale("log")
plt.xlabel("n_estimators (log scale)")
plt.ylabel("Bias")
plt.title("Bias vs Ensemble Size")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

plt.tight_layout()
plt.savefig("bias_vs_ensemble.pdf")
plt.savefig("bias_vs_ensemble.png", dpi=300)
plt.show()
