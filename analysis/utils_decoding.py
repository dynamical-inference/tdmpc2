from itertools import combinations
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, MultiTaskLasso
from sklearn.metrics import r2_score
from tqdm import tqdm


def standardize_fit(X, eps=1e-8):
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + eps
    return mu, sd


def standardize_apply(X, mu, sd):
    return (X - mu) / sd


def make_lagged_pairs_for_episode(x_ep, y_ep, lag, predict_difference=False):
    """
    Build (X, Y) pairs within one episode for given lag.
    
    Args:
        x_ep: [T, L] array of input features
        y_ep: [T, D] array of targets
        lag: integer lag (positive = predict future, negative = predict past)
        predict_difference: if True, target is y_{t+lag} - y_t instead of y_{t+lag}
    
    Returns:
        X: [N, L] array of inputs (x_t)
        Y: [N, D] array of targets (y_{t+lag} or y_{t+lag} - y_t)
    """
    T = len(x_ep)
    if lag >= 0:
        t_src = np.arange(0, T - lag)
        t_tgt = t_src + lag
    else:
        # Predict the past: (z_t, y_{t+lag}) = (z_t, y_{t-|lag|})
        shift = abs(lag)
        t_src = np.arange(shift, T)
        t_tgt = np.arange(0, T - shift)

    X = x_ep[t_src]
    Y = y_ep[t_tgt]

    if predict_difference and lag != 0:
        # Target becomes the change: y_{t+lag} - y_t
        Y = Y - y_ep[t_src]

    return X, Y


def bootstrap_ci(vals, alpha=0.05, n_boot=1000):
    rng = np.random.RandomState(0)
    boot_means = np.zeros(n_boot)
    for b in range(n_boot):
        resampled = rng.choice(vals, size=len(vals), replace=True)
        boot_means[b] = np.mean(resampled)

    # NOTE: np.percentile expects the percentage to be in the range [0, 100]
    # so we multiply by 100 and divide by 2 to get the correct percentage.
    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return (lower, upper)


def evaluate_lags_leave_k_out(x_eps,
                              y_eps,
                              lags,
                              k_holdout=1,
                              model='linear_regression',
                              max_combinations=None,
                              add_bootstrap_ci=False,
                              alpha=0.05,
                              n_bootstrap=1000,
                              shuffle_X=False,
                              shuffle_y=False,
                              predict_difference=False,
                              standardize_X=False,
                              standardize_Y=False,
                              standardize_eps=1e-8):
    """
    Episode-wise CV decoding with optional train-only standardization.

    Args:
        x_eps: list of [T, L] arrays (inputs, e.g. activations z_t)
        y_eps: list of [T, D] arrays (targets, e.g. state y_t)
        lags: iterable of int lags
        k_holdout: number of held-out episodes per fold
        model: 'linear_regression' | 'lasso' | 'multi_task_lasso'
        standardize_X: if True, z-score X using TRAIN stats per (fold, lag)
        standardize_Y: if True, z-score Y using TRAIN stats per (fold, lag) (R² unchanged in theory, but helps Lasso)
        predict_difference: if True, target is y_{t+lag} - y_t (for lag != 0)
    """
    assert isinstance(x_eps, list) and isinstance(
        y_eps, list), "x_eps and y_eps must be lists"
    n_eps = len(x_eps)
    assert n_eps == len(y_eps), "x_eps and y_eps must have same length"
    assert n_eps > k_holdout, "k_holdout must be smaller than number of episodes"

    # Enumerate folds (all combinations or sampled subset)
    all_combos = list(combinations(range(n_eps), k_holdout))
    if max_combinations is not None and len(all_combos) > max_combinations:
        rng = np.random.default_rng(0)
        sel = rng.choice(len(all_combos), size=max_combinations, replace=False)
        all_combos = [all_combos[i] for i in sel]

    if shuffle_X:
        rng = np.random.default_rng(0)
        x_eps = [x_ep[rng.permutation(len(x_ep))] for x_ep in x_eps]
    if shuffle_y:
        rng = np.random.default_rng(0)
        y_eps = [y_ep[rng.permutation(len(y_ep))] for y_ep in y_eps]

    r2_by_lag = {lag: [] for lag in lags}
    model_by_lag = {lag: [] for lag in lags}

    for test_idxs in tqdm(all_combos):
        test_idxs = np.atleast_1d(test_idxs)
        train_idxs = [i for i in range(n_eps) if i not in set(test_idxs)]

        for lag in lags:
            # Collect training pairs (episode-local lagging, then concat)
            X_train_list, Y_train_list = [], []
            for i in train_idxs:
                X_i, Y_i = make_lagged_pairs_for_episode(
                    x_eps[i],
                    y_eps[i],
                    lag,
                    predict_difference=predict_difference)
                X_train_list.append(X_i)
                Y_train_list.append(Y_i)

            X_train = np.concatenate(X_train_list, axis=0)
            Y_train = np.concatenate(Y_train_list, axis=0)

            # Fit standardization on TRAIN only
            if standardize_X:
                X_mu, X_sd = standardize_fit(X_train, eps=standardize_eps)
                X_train_use = standardize_apply(X_train, X_mu, X_sd)
            else:
                X_mu = X_sd = None
                X_train_use = X_train

            if standardize_Y:
                Y_mu, Y_sd = standardize_fit(Y_train, eps=standardize_eps)
                Y_train_use = standardize_apply(Y_train, Y_mu, Y_sd)
            else:
                Y_mu = Y_sd = None
                Y_train_use = Y_train

            # Choose model
            if model == 'linear_regression':
                model_to_fit = LinearRegression()
            elif model == 'lasso':
                model_to_fit = Lasso(alpha=0.001)
            elif model == 'multi_task_lasso':
                model_to_fit = MultiTaskLasso(alpha=0.01)
            else:
                raise ValueError(f"Model {model} not supported")

            # Fit model on training set
            model_to_fit.fit(X_train_use, Y_train_use)

            # Evaluate on held-out episodes
            fold_r2s = []
            for j in test_idxs:
                X_te, Y_te = make_lagged_pairs_for_episode(
                    x_eps[j],
                    y_eps[j],
                    lag,
                    predict_difference=predict_difference)

                if standardize_X:
                    X_te_use = standardize_apply(X_te, X_mu, X_sd)
                else:
                    X_te_use = X_te

                # Predict in standardized Y space if requested, then invert
                Y_pred_use = model_to_fit.predict(X_te_use)
                if standardize_Y:
                    Y_pred = Y_pred_use * Y_sd + Y_mu
                else:
                    Y_pred = Y_pred_use

                r2 = r2_score(Y_te, Y_pred, multioutput='variance_weighted')
                fold_r2s.append(r2)

            r2_by_lag[lag].append(float(np.mean(fold_r2s)))
            model_by_lag[lag].append(model_to_fit)

    r2_mean = {lag: float(np.mean(vals)) for lag, vals in r2_by_lag.items()}
    r2_std = {
        lag: float(np.std(vals, ddof=1)) for lag, vals in r2_by_lag.items()
    }

    results = {
        'r2_mean': r2_mean,
        'r2_std': r2_std,
        'r2_by_lag': r2_by_lag,
        'model_by_lag': model_by_lag,
    }

    if add_bootstrap_ci:
        results['r2_ci'] = {
            lag: bootstrap_ci(vals, alpha=alpha, n_boot=n_bootstrap)
            for lag, vals in r2_by_lag.items()
        }

    return results
