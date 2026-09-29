# Paste this entire file into one notebook cell.
import time
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.linalg import subspace_angles
import utils_data

DEFAULT_CONFIG = {
    'dataset': 'MNIST',
    'root': '../../data',
    'classes': [0],
    'limit_n_samples': 1000,  # Total across selected classes; None keeps all.
    'samples_per_class': None,  # Exact count per class; disable the total cap to keep balance.
    'latent_dimension': 2,
    'noise_level': 0.0,      # Generation std = this multiplier * fitted sigma.
    'n_generated': 10,
    'em_iterations': 5000,
    'elbo_iterations': 5000,
    'learning_rate': 1e-3,
    'seed': 0,
    'sigma_min': 1e-4,       # Explicit constraint shared by all three fits.
    'metric_every': 50,     # Expensive metrics at 1, every 50, and final iteration.
    'show_plots': True,
}
_DATA_CACHE = {}


def prepare_data(config):
    key = (config['dataset'], str(Path(config['root']).resolve()), config['seed'])
    if key not in _DATA_CACHE:
        # utils_data shuffles; isolate its random state and cache the loaded data.
        with torch.random.fork_rng():
            torch.manual_seed(config['seed'])
            _DATA_CACHE[key] = utils_data.load_dataset(key[0], root=key[1])
    X_all, y_all = _DATA_CACHE[key]
    classes = list(dict.fromkeys(config['classes']))
    if not classes:
        raise ValueError('Select at least one class.')
    mask = torch.zeros_like(y_all, dtype=torch.bool)
    for label in classes:
        if not (y_all == label).any():
            raise ValueError(f'Class {label} is absent from this dataset.')
        mask |= y_all == label
    X, y = X_all[mask], y_all[mask]
    generator = torch.Generator().manual_seed(config['seed'])
    per_class = config.get('samples_per_class')
    if per_class is not None:
        if not isinstance(per_class, int) or per_class < 1:
            raise ValueError('samples_per_class must be a positive integer or None.')
        if config['limit_n_samples'] is not None:
            raise ValueError('Set limit_n_samples=None when using samples_per_class.')
        selected = []
        for label in classes:
            indices = torch.where(y == label)[0]
            if len(indices) < per_class:
                raise ValueError(f'Class {label} has only {len(indices)} images; requested {per_class}.')
            # Each class keeps the same subset across different class combinations.
            class_generator = torch.Generator().manual_seed(config['seed'] + int(label))
            permutation = torch.randperm(len(indices), generator=class_generator)
            selected.append(indices[permutation[:per_class]])
        selected = torch.cat(selected)
        X, y = X[selected], y[selected]
    order = torch.randperm(len(X), generator=generator)
    k = config['limit_n_samples']
    if k is not None:
        if not isinstance(k, int) or k < 1:
            raise ValueError('limit_n_samples must be a positive integer or None.')
        if k > len(X):
            print(f'Requested {k} images; using all {len(X)} available images.')
        order = order[:k]
    return X[order].double(), y[order]


def compute_sample_mean_cov(X):
    mean = X.mean(0)
    Y = X - mean
    return mean, Y.T @ Y / len(X)


def fit_closed_form(X, d, sigma_min):
    start = time.perf_counter()
    b, cov = compute_sample_mean_cov(X)
    values, U = torch.linalg.eigh(cov)
    values, U = values.flip(0).clamp_min(0), U.flip(1)
    raw_variance = values[d:].mean()
    variance = raw_variance.clamp_min(sigma_min**2)
    W = U[:, :d] @ torch.diag((values[:d] - variance).clamp_min(0).sqrt())
    return {'W': W, 'b': b, 'sigma': variance.sqrt(),
            'seconds': time.perf_counter() - start,
            'floor_active': bool(raw_variance < sigma_min**2)}


def model_covariance(model):
    W, sigma = model['W'], model['sigma']
    return W @ W.T + sigma.square() * torch.eye(W.shape[0], dtype=W.dtype)


def log_likelihood(X, b, W, sigma):
    N, D = X.shape
    C = W @ W.T + sigma.square() * torch.eye(D, dtype=X.dtype)
    L = torch.linalg.cholesky(C)
    Y = (X - b).T
    quadratic = (Y * torch.cholesky_solve(Y, L)).sum()
    return -0.5 * (N * (D * np.log(2 * np.pi)
                       + 2 * L.diagonal().log().sum()) + quadratic)


def elbo(X, W, b, sigma, A, c, E):
    N, D = X.shape
    d = W.shape[1]
    mu = X @ A.T + c
    residual = X - mu @ W.T - b
    Sigma_e = E @ E.T
    reconstruction = -N * D / 2 * torch.log(2 * torch.pi * sigma.square())
    reconstruction -= (residual.square().sum()
                       + N * torch.trace((W.T @ W) @ Sigma_e)) / (2 * sigma.square())
    kl = 0.5 * (N * torch.trace(Sigma_e) + mu.square().sum() - N * d
                - N * torch.linalg.slogdet(Sigma_e)[1])
    return reconstruction - kl


def gaussian_kl(mu_p, Sigma_p, mu_q, Sigma_q):
    diff = mu_p - mu_q
    logdet_p = torch.linalg.slogdet(Sigma_p)[1]
    logdet_q = torch.linalg.slogdet(Sigma_q)[1]
    return (0.5 * (logdet_q - logdet_p - mu_p.numel()
                  + torch.trace(torch.linalg.solve(Sigma_q, Sigma_p))
                  + diff @ torch.linalg.solve(Sigma_q, diff))).item()


def subspace_error(W, W_ref):
    # A d-dimensional reference is not identifiable if its loading rank is < d.
    if torch.linalg.matrix_rank(W_ref) < W_ref.shape[1]:
        return float('nan')
    if torch.linalg.matrix_rank(W) < W.shape[1]:
        return float('nan')
    return float(np.degrees(subspace_angles(W.detach().numpy(), W_ref.numpy()).max()))


@torch.no_grad()
def record_metrics(history, iteration, seconds, X, model, reference, C_ref):
    W, b, sigma = (model[k] for k in ('W', 'b', 'sigma'))
    history.append({
        'iteration': iteration, 'seconds': seconds,
        'll': log_likelihood(X, b, W, sigma).item(),
        'kl': gaussian_kl(reference['b'], C_ref, b, model_covariance(model)),
        'angle': subspace_error(W, reference['W']),
        'b_error': ((b - reference['b']).norm() /
                    reference['b'].norm().clamp_min(1e-12)).item(),
        'sigma_error': ((sigma - reference['sigma']).abs() / reference['sigma']).item(),
    })


def fit_iterative(X, d, config, reference, method):
    N, D = X.shape
    W = torch.eye(D, dtype=X.dtype)[:, :d].clone()
    b = torch.zeros(D, dtype=X.dtype)
    sigma = X.new_tensor(0.1)
    history, seconds = [], 0.0
    C_ref = model_covariance(reference)
    iterations = config['em_iterations' if method == 'EM' else 'elbo_iterations']
    if method == 'ELBO':
        A = W.T.clone()
        c = torch.zeros(d, dtype=X.dtype)
        E = torch.eye(d, dtype=X.dtype)
        parameters = [W, b, sigma, A, c, E]
        for parameter in parameters:
            parameter.requires_grad_(True)
        optimizer = torch.optim.Adam(parameters, lr=config['learning_rate'])
    for iteration in range(1, iterations + 1):
        start = time.perf_counter()
        if method == 'EM':
            M = W.T @ W + sigma.square() * torch.eye(d, dtype=X.dtype)
            M_inv = torch.linalg.inv(M)
            Ez = (X - b) @ W @ M_inv
            EzzT = N * sigma.square() * M_inv + Ez.T @ Ez
            # Joint mean/loading M-step, consistent with the zero initialization.
            mean_x, mean_z = X.mean(0), Ez.mean(0)
            centered_second = EzzT - N * torch.outer(mean_z, mean_z)
            numerator = (X - mean_x).T @ Ez
            W = torch.linalg.solve(centered_second, numerator.T).T
            b = mean_x - W @ mean_z
            Y = X - b
            variance = (Y.square().sum() - 2 * (Y * (Ez @ W.T)).sum()
                        + torch.trace(W.T @ W @ EzzT)) / (N * D)
            sigma = variance.clamp_min(config['sigma_min']**2).sqrt()
        else:
            optimizer.zero_grad(set_to_none=True)
            loss = -elbo(X, W, b, sigma, A, c, E)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite ELBO. Try a smaller learning_rate.')
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                sigma.clamp_(min=config['sigma_min'])
        seconds += time.perf_counter() - start
        if iteration == 1 or iteration % config['metric_every'] == 0 or iteration == iterations:
            record_metrics(history, iteration, seconds, X,
                           {'W': W, 'b': b, 'sigma': sigma}, reference, C_ref)
    result = {k: v.detach().clone() for k, v in {'W': W, 'b': b, 'sigma': sigma}.items()}
    result.update(history=history, seconds=seconds)
    if method == 'ELBO':
        result.update({k: v.detach().clone() for k, v in {'A': A, 'c': c, 'E': E}.items()})
    return result


@torch.no_grad()
def sanity_check(X, models):
    ref = models['Closed-form']
    W, b, sigma = (ref[k] for k in ('W', 'b', 'sigma'))
    M = W.T @ W + sigma.square() * torch.eye(W.shape[1], dtype=X.dtype)
    A = torch.linalg.solve(M, W.T)
    c = -A @ b
    covariance = sigma.square() * torch.linalg.inv(M)
    E = torch.linalg.cholesky((covariance + covariance.T) / 2)
    print('\nSanity checks (summed objectives):')
    for name, model in models.items():
        ll = log_likelihood(X, model['b'], model['W'], model['sigma']).item()
        print(f'{name:12s} log-likelihood: {ll:.6f}; per sample: {ll / len(X):.6f}')
    ref_elbo = elbo(X, W, b, sigma, A, c, E).item()
    ref_ll = log_likelihood(X, b, W, sigma).item()
    print(f'Reference exact-posterior ELBO: {ref_elbo:.6f}; LL gap: {ref_ll-ref_elbo:.3g}')
    learned = models['ELBO']
    learned_elbo = elbo(X, **{k: learned[k] for k in ('W', 'b', 'sigma', 'A', 'c', 'E')}).item()
    print(f'Learned encoder ELBO: {learned_elbo:.6f}')
    return ref_ll


def show_data(X):
    mean, cov = compute_sample_mean_cov(X)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].imshow(mean.reshape(28, 28), cmap='gray', vmin=-1, vmax=1)
    axes[0].set_title('Sample mean')
    image = axes[1].imshow(cov, cmap='viridis')
    axes[1].set_title('Sample covariance'); fig.colorbar(image, ax=axes[1])
    plt.tight_layout(); plt.show()


@torch.no_grad()
def sample_models(result, noise_level=None, n_generated=None, show=True):
    config = result['config']
    scale = config['noise_level'] if noise_level is None else noise_level
    n = config['n_generated'] if n_generated is None else n_generated
    if scale < 0 or n < 1:
        raise ValueError('noise_level must be nonnegative and n_generated positive.')
    d = config['latent_dimension']
    D = result['X'].shape[1]
    generator = torch.Generator().manual_seed(config['seed'] + 1)
    # Reuse draws when changing noise level for a controlled visual comparison.
    z = torch.randn(n, d, generator=generator, dtype=result['X'].dtype)
    eps = torch.randn(n, D, generator=generator, dtype=result['X'].dtype)
    samples = {}
    if show:
        fig, axes = plt.subplots(3, n, figsize=(2*n, 6), squeeze=False)
    for row, (name, model) in enumerate(result['models'].items()):
        samples[name] = model['b'] + z @ model['W'].T + scale * model['sigma'] * eps
        if show:
            for j, ax in enumerate(axes[row]):
                ax.imshow(samples[name][j].reshape(28, 28), cmap='gray', vmin=-1, vmax=1)
                ax.set_xticks([]); ax.set_yticks([])
                if j == 0:
                    ax.set_ylabel(name)
    if show:
        fig.suptitle(f"{config['dataset']} | d={d} | N={len(result['X'])} | noise × {scale}")
        plt.tight_layout(); plt.show()
    return samples


def plot_evaluation(result):
    models = result['models']
    panels = [('ll', 'Objective'),
              ('kl', 'KL to closed-form'),
              ('angle', 'W subspace error (deg)'),
              ('b_error', 'b relative error'),
              ('sigma_error', 'sigma relative error'),
              ('seconds', 'Cumulative wall-clock (s)')]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (key, title) in zip(axes.ravel(), panels):
        for name in ('EM', 'ELBO'):
            history = models[name]['history']
            ax.plot([h['iteration'] for h in history], [h[key] for h in history], label=name)
        ax.set_title(title)
        ax.legend()
    plt.tight_layout(); plt.show()


def run_experiment(config):
    config = {**DEFAULT_CONFIG, **config}
    if config['sigma_min'] <= 0 or config['sigma_min'] > 0.1:
        raise ValueError('Choose 0 < sigma_min <= 0.1 (the initial sigma).')
    for key in ('em_iterations', 'elbo_iterations', 'metric_every', 'n_generated'):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer.')
    if config['learning_rate'] <= 0 or config['noise_level'] < 0:
        raise ValueError('learning_rate must be positive and noise_level nonnegative.')
    X, y = prepare_data(config)
    d = config['latent_dimension']
    if not isinstance(d, int) or not 0 < d < X.shape[1]:
        raise ValueError('latent_dimension must be an integer between 1 and D-1.')
    labels, counts = y.unique(return_counts=True)
    print(f"\n{config['dataset']} | N={len(X)}, D={X.shape[1]}, d={d}")
    print('Class counts:', dict(zip(labels.tolist(), counts.tolist())))
    ref = fit_closed_form(X, d, config['sigma_min'])
    if ref['floor_active']:
        print('Noise floor active: reference is a constrained fit, not a finite unconstrained MLE.')
    models = {'Closed-form': ref}
    for method in ('EM', 'ELBO'):
        print(f'Fitting {method}...')
        models[method] = fit_iterative(X, d, config, ref, method)
    result = {'config': config, 'X': X, 'y': y, 'models': models}
    result['reference_ll'] = sanity_check(X, models)
    if config['show_plots']:
        show_data(X)
    result['samples'] = sample_models(result, show=config['show_plots'])
    if config['show_plots']:
        plot_evaluation(result)
    return result
