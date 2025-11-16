from itertools import combinations
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, MultiTaskLasso
from sklearn.metrics import r2_score
from tqdm import tqdm


def make_lagged_pairs_for_episode(x_ep, y_ep, lag):
    """Build (X, Y) pairs within one episode for given lag."""
    T = len(x_ep)
    if lag >= 0:
        t_src = np.arange(0, T - lag)
        t_tgt = t_src + lag

    elif lag < 0:
        # Predict the past: (z_t, y_{t+lag}) = (z_t, y_{t-|lag|})
        shift = abs(lag)
        t_src = np.arange(shift, T)
        t_tgt = np.arange(0, T - shift)

    return x_ep[t_src], y_ep[t_tgt]


def bootstrap_ci(vals, alpha=0.05, n_boot=1000):
    rng = np.random.RandomState(0)
    boot_means = np.zeros(n_boot)
    for b in range(n_boot):
        resampled = rng.choice(vals, size=len(vals), replace=True)
        boot_means[b] = np.mean(resampled)

    lower = np.percentile(boot_means, alpha)
    upper = np.percentile(boot_means, 100 - alpha)
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
                              shuffle_y=False):
    """
    Args:
        x_eps: list of [N_eps_i, L] arrays
        y_eps: list of [N_eps_i, D] arrays   
    """
    n_eps = len(x_eps)
    assert n_eps > k_holdout, "k_holdout must be smaller than number of episodes"

    # Enumerate folds (all combinations or sampled subset)
    all_combos = list(combinations(range(n_eps), k_holdout))
    if max_combinations is not None and len(all_combos) > max_combinations:
        np.random.seed(0)
        all_combos = list(
            np.random.choice(len(all_combos), max_combinations, replace=False))

    if shuffle_X:
        x_eps = [np.random.permutation(x_ep) for x_ep in x_eps]
    if shuffle_y:
        y_eps = [np.random.permutation(y_ep) for y_ep in y_eps]

    r2_by_lag = {lag: [] for lag in lags}
    model_by_lag = {lag: [] for lag in lags}

    for test_idxs in tqdm(all_combos):
        test_idxs = np.atleast_1d(test_idxs)
        train_idxs = [i for i in range(n_eps) if i not in test_idxs]

        for lag in lags:
            # Collect training pairs
            X_train_list, Y_train_list = [], []
            for i in train_idxs:
                X_i, Y_i = make_lagged_pairs_for_episode(
                    x_eps[i], y_eps[i], lag)
                X_train_list.append(X_i)
                Y_train_list.append(Y_i)

            X_train = np.concatenate(X_train_list, axis=0)
            Y_train = np.concatenate(Y_train_list, axis=0)

            if model == 'linear_regression':
                model_to_fit = LinearRegression()
            elif model == 'lasso':
                model_to_fit = Lasso(alpha=0.001)
            # elif model == 'lasso_cv':
            #     from sklearn.linear_model import MultiTaskLasso
            #     model_to_fit = MultiTaskLassoCV(cv=5,
            #                                     n_alphas=20,
            #                                     random_state=123)
            elif model == 'multi_task_lasso':
                model_to_fit = MultiTaskLasso(alpha=0.01)
            else:
                raise ValueError(f"Model {model} not supported")

            # Fit model on training set
            model_to_fit = model_to_fit.fit(X_train, Y_train)

            # Evaluate on each held-out episode
            fold_r2s = []
            for j in test_idxs:
                X_te, Y_te = make_lagged_pairs_for_episode(
                    x_eps[j], y_eps[j], lag)
                Y_pred = model_to_fit.predict(X_te)
                r2 = r2_score(Y_te, Y_pred, multioutput='variance_weighted')
                fold_r2s.append(r2)

            r2_by_lag[lag].append(
                np.mean(fold_r2s))  # mean R² over held-out eps in this fold

            model_by_lag[lag].append(model_to_fit)

    # Aggregate mean and std over folds
    r2_mean = {lag: np.mean(vals) for lag, vals in r2_by_lag.items()}
    r2_std = {lag: np.std(vals, ddof=1) for lag, vals in r2_by_lag.items()}

    results = {
        'r2_mean': r2_mean,
        'r2_std': r2_std,
        'r2_by_lag': r2_by_lag,
        'model_by_lag': model_by_lag,
    }

    if add_bootstrap_ci:
        r2_ci = {
            lag: bootstrap_ci(vals, alpha=alpha, n_boot=n_bootstrap)
            for lag, vals in r2_by_lag.items()
        }
        results['r2_ci'] = r2_ci

    return results
