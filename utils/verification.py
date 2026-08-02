# -----------------------------------------------------------------------------------------
# -------------------------- Model Verification -------------------------------------------
# -----------------------------------------------------------------------------------------

import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    recall_score, precision_score, f1_score,
    roc_auc_score, average_precision_score
)

# ── Cost matrix (SCANIA Component X) ──────────────────────────────────────────
SCANIA_COST = {
    (0, 0): 0,   (0, 1): 7,   (0, 2): 8,   (0, 3): 9,   (0, 4): 10,
    (1, 0): 200, (1, 1): 0,   (1, 2): 7,   (1, 3): 8,   (1, 4): 9,
    (2, 0): 300, (2, 1): 200, (2, 2): 0,   (2, 3): 7,   (2, 4): 8,
    (3, 0): 400, (3, 1): 300, (3, 2): 200, (3, 3): 0,   (3, 4): 7,
    (4, 0): 500, (4, 1): 400, (4, 2): 300, (4, 3): 200, (4, 4): 0,
}
 
# Predict-all-zero baseline cost/vehicle — used as quality filter floor
BASELINE_COST = 8.62
 
# Threshold grid for cost-optimal search (same as model_evaluation.ipynb)
THRESHOLD_GRID = np.arange(0.01, 1.0, 0.01)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────
 
def _set_seed(seed: int):
    """Set ALL sources of randomness so each run is independently reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
 
 
def _compute_cost_per_vehicle(y_true_mc, probs, threshold):
    """Total SCANIA cost / vehicle at a given threshold (binary → class 4 mapping)."""
    y_pred = (probs >= threshold).astype(int) * 4
    total  = sum(SCANIA_COST[(int(yt), int(yp))] for yt, yp in zip(y_true_mc, y_pred))
    return total / len(y_true_mc)
 
 
def _find_optimal_threshold(y_true_mc, probs):
    """Grid search over THRESHOLD_GRID; returns (best_threshold, best_cost_per_veh)."""
    costs   = [_compute_cost_per_vehicle(y_true_mc, probs, t) for t in THRESHOLD_GRID]
    best_idx = int(np.argmin(costs))
    return THRESHOLD_GRID[best_idx], costs[best_idx]
 
 
def _smooth(values: list, window: int = 3) -> np.ndarray:
    """Simple rolling average to reduce oscillation noise before slope checks."""
    return np.convolve(values, np.ones(window) / window, mode='valid')
 
 
def _passes_filter(history: dict, val_auc_pr: float, val_cost: float):
    """
    Returns (passed: bool, reason: str).
 
    Six checks calibrated to observed curve values in this project:
      - Healthy val loss converges to ~0.024-0.026
      - Healthy train loss converges to ~0.034-0.036
      - Train/val gap is small and stable in healthy runs
      - Steepest learning always happens in first 20% of epochs
 
    Checks:
      1. Peak val AUC-PR floor     — must exceed random chance (2.6% pos rate)
      2. Val cost floor            — must beat predict-all-zero baseline
      3. Final val loss ceiling    — healthy runs end ≤ 0.028; above = failed convergence
      4. Train/val divergence gap  — val - train gap must stay < 0.008; wider = overfit
      5. Level contract            — mean(w1) > mean(w2) > mean(w3) on train loss
      6. Slope contract            — |slope_w1| > |slope_w2| and |slope_w1| > |slope_w3|
                                     steepest learning was early (on smoothed train loss)
    """
    MIN_AUC_PR      = 0.030   # just above random chance for 2.6% positive rate
    MAX_FINAL_VAL   = 0.028   # healthy runs end around 0.024–0.026; above this = poor convergence
    MAX_TRAINVAL_GAP= 0.008   # train/val gap in healthy runs is ~0.002–0.005
 
    train_loss = history['train_loss']
    val_loss   = history['val_loss']
    n          = len(train_loss)
 
    # ── Check 1: Peak val AUC-PR floor ───────────────────────────────────────
    peak_auc_pr = max(history['val_auc_pr'])
    if peak_auc_pr < MIN_AUC_PR:
        return False, f"Peak val AUC-PR {peak_auc_pr:.4f} < {MIN_AUC_PR} (model learned nothing)"
 
    # ── Check 2: Val cost floor ───────────────────────────────────────────────
    if val_cost >= BASELINE_COST:
        return False, f"Val cost/veh {val_cost:.3f} >= {BASELINE_COST} (worse than predict-all-0)"
 
    # ── Check 3: Final val loss ceiling ──────────────────────────────────────
    final_val_loss = val_loss[-1]
    if final_val_loss > MAX_FINAL_VAL:
        return False, (
            f"Final val loss {final_val_loss:.4f} > {MAX_FINAL_VAL} "
            f"(failed to converge — healthy runs end ≤ {MAX_FINAL_VAL})"
        )
    
    improvement = val_loss[0] - val_loss[-1]
    noise = np.std(val_loss)  

    snr = improvement / noise  # signal-to-noise ratio
    if snr < 1.0:
        return False, f"Signal-to-noise ratio {snr:.2f} < 1.0 -> val loss too noisy, no clear learning trend"
    
    if improvement < 0.002:
        return False, f"Val loss improvement {improvement:.4f} < 0.002 — model did not learn"

 
    # ── Check 4: Train/val divergence gap ────────────────────────────────────
    final_gap = val_loss[-1] - train_loss[-1]
    if final_gap > MAX_TRAINVAL_GAP:
        return False, (
            f"Train/val gap {final_gap:.4f} > {MAX_TRAINVAL_GAP} "
            f"(val loss diverging from train loss — overfitting)"
        )
 
    # ── Need enough epochs for window checks ──────────────────────────────────
    if n < 10:
        return False, f"Only {n} epochs — too short to assess training curve shape"
 
    # ── Define three windows on train loss ────────────────────────────────────
    w1_end = max(1,        int(n * 0.20))
    w2_end = max(w1_end+1, int(n * 0.80))
 
    window1 = train_loss[:w1_end]
    window2 = train_loss[w1_end:w2_end]
    window3 = train_loss[w2_end:]
 
    if len(window2) < 3 or len(window3) < 3:
        return True, "OK (too short for shape checks — passed checks 1-4)"
 
    mean1 = np.mean(window1)
    mean2 = np.mean(window2)
    mean3 = np.mean(window3)
 
    # ── Check 5: Level contract ───────────────────────────────────────────────
    if not (mean1 > mean2):
        return False, (
            f"Level contract failed: mean(w1)={mean1:.4f} not > mean(w2)={mean2:.4f} "
            f"— loss never improved from early to mid training"
        )
    if not (mean2 > mean3):
        return False, (
            f"Level contract failed: mean(w2)={mean2:.4f} not > mean(w3)={mean3:.4f} "
            f"— smiley shape detected (loss rose in final window)"
        )
 
    # ── Check 6: Slope contract on smoothed windows ───────────────────────────
    try:
        s1 = _smooth(window1) if len(window1) >= 3 else np.array(window1)
        s2 = _smooth(window2) if len(window2) >= 3 else np.array(window2)
        s3 = _smooth(window3) if len(window3) >= 3 else np.array(window3)

        slope1 = np.polyfit(range(len(s1)), s1, deg=1)[0]
        slope2 = np.polyfit(range(len(s2)), s2, deg=1)[0]
        slope3 = np.polyfit(range(len(s3)), s3, deg=1)[0]

    except (np.linalg.LinAlgError, ValueError):
        return True, "OK (slope check skipped — short window)"

    if not (abs(slope1) > abs(slope2)):
        return False, (
            f"Slope contract failed: |slope_w1|={abs(slope1):.6f} not > "
            f"|slope_w2|={abs(slope2):.6f} — learning did not slow down mid-training"
        )
    if not (abs(slope1) > abs(slope3)):
        return False, (
            f"Slope contract failed: |slope_w1|={abs(slope1):.6f} not > "
            f"|slope_w3|={abs(slope3):.6f} — late training as steep as early "
            f"(oscillating or diverging)"
        )

    return True, "OK"
 
 
def _evaluate_run(trainer, X_val, X_test, val_labels_mc, test_labels_mc):
    """
    Get probabilities from trainer, find optimal threshold on val,
    then compute all metrics on both val and test.
    Returns a metrics dict.
    """
    _, probs_val  = trainer.predict(X_val)
    _, probs_test = trainer.predict(X_test)
 
    opt_threshold, val_cost = _find_optimal_threshold(val_labels_mc, probs_val)
 
    y_val_bin  = (val_labels_mc  > 0).astype(int)
    y_test_bin = (test_labels_mc > 0).astype(int)
 
    val_pred  = (probs_val  >= opt_threshold).astype(int)
    test_pred = (probs_test >= opt_threshold).astype(int)
 
    return {
        # Val
        'val_auc_pr'       : average_precision_score(y_val_bin, probs_val),
        'val_auc_roc'      : roc_auc_score(y_val_bin, probs_val),
        'val_recall'       : recall_score(y_val_bin,  val_pred,  zero_division=0),
        'val_precision'    : precision_score(y_val_bin, val_pred, zero_division=0),
        'val_f1'           : f1_score(y_val_bin, val_pred, zero_division=0),
        'val_cost_per_veh' : val_cost,
        # Test
        'test_auc_pr'      : average_precision_score(y_test_bin, probs_test),
        'test_auc_roc'     : roc_auc_score(y_test_bin, probs_test),
        'test_recall'      : recall_score(y_test_bin,  test_pred, zero_division=0),
        'test_precision'   : precision_score(y_test_bin, test_pred, zero_division=0),
        'test_f1'          : f1_score(y_test_bin, test_pred, zero_division=0),
        'test_cost_per_veh': _compute_cost_per_vehicle(test_labels_mc, probs_test, opt_threshold),
        # Threshold
        'optimal_threshold': float(opt_threshold),
        # Raw probs (useful for plotting PR curves later)
        'probs_val'        : probs_val,
        'probs_test'       : probs_test,
    }
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
 
def run_verification(
    model_name:    str,
    hp:            dict,
    seeds:         list,
    utils_module,           # utils module (passed in to avoid circular imports)
    X_train,  y_train,
    X_val,    y_val,
    X_test,
    val_labels_mc,          # multiclass labels for val  (for cost evaluation)
    test_labels_mc,         # multiclass labels for test (for cost evaluation)
    model_dir,              # Path where to save best checkpoint per run
):
    """
    Train a model N times across different seeds and return aggregated results.
 
    Args:
        model_name:    'LSTM' or 'TCN' (case-insensitive)
        hp:            loaded hyperparams.json dict
        seeds:         list of ints, e.g. [42, 123, 7, 99, 2024]
        utils_module:  your imported utils (passed in so this file stays standalone)
        X_train/val/test, y_train/val: numpy arrays from generate_sequential_data
        val_labels_mc / test_labels_mc: multiclass label arrays aligned to vehicle order
        model_dir:     Path to save .pt checkpoints
 
    Returns:
        dict with keys:
            'passed'   — list of run dicts that passed the quality filter
            'rejected' — list of run dicts that were rejected, with reason
            'summary'  — mean ± std over passing runs for all metrics
    """
 
    # ── Model registry ────────────────────────────────────────────────────────
    MODEL_REGISTRY = {
        'lstm': utils_module.LSTMModel,
        'tcn' : utils_module.TCNModel,
    }
 
    model_cls = MODEL_REGISTRY.get(model_name.lower())
    if model_cls is None:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from: {list(MODEL_REGISTRY.keys())}"
        )
 
    # ── Model kwargs from hp dict ─────────────────────────────────────────────
    # Each model type has its own key in hyperparams.json
    model_key = model_name.upper()   # 'LSTM' or 'TCN'
    model_hp  = hp[model_key]
    input_size = hp['INPUT_SIZE']
 
    if model_name.lower() == 'lstm':
        model_kwargs = dict(
            input_size  = input_size,
            hidden_size = model_hp['hidden_size'],
            num_layers  = model_hp['num_layers'],
            dropout     = model_hp['dropout'],
        )
        trainer_kwargs = dict(
            lr        = model_hp['learning_rate'],
            batch_size= hp['LSTM'].get('batch_size', 64),
            clip_grad = model_hp.get('clip_grad', 1.0),
        )
    else:  # TCN
        model_kwargs = dict(
            input_size   = input_size,
            num_channels = model_hp['num_channels'],
            num_layers   = model_hp['num_layers'],
            kernel_size  = model_hp['kernel_size'],
            dropout      = model_hp['dropout'],
        )
        trainer_kwargs = dict(
            lr        = model_hp['learning_rate'],
            batch_size= hp['TCN'].get('batch_size', 128),
            clip_grad = None,
        )
 
    # ── Training epochs / patience (use defaults if not in hp) ───────────────
    num_epochs = hp.get('NUM_EPOCHS', 1000)
    patience   = model_hp.get('patience', 15 if model_name.lower() == 'lstm' else 10)
 
    # ── Run loop ──────────────────────────────────────────────────────────────
    passed   = []
    rejected = []
 
    for seed in seeds:
        print(f"\n{'='*55}")
        print(f"  {model_name} | seed = {seed}")
        print(f"{'='*55}")
 
        # 1. Seed everything before model construction
        _set_seed(seed)
 
        # 2. Fresh model and trainer every iteration
        model   = model_cls(**model_kwargs)
        trainer = utils_module.BaseTrainer(model=model, **trainer_kwargs)
 
        # 3. Train
        save_path = model_dir / f'{model_name.lower()}_seed{seed}_best.pt'
        history   = trainer.fit(
            X_train, y_train,
            X_val,   y_val,
            num_epochs = num_epochs,
            patience   = patience,
            save_path  = str(save_path),
        )
 
        # 4. Evaluate — threshold search on val, metrics on val + test
        metrics = _evaluate_run(
            trainer, X_val, X_test, val_labels_mc, test_labels_mc
        )
 
        # 5. Quality filter
        passed_filter, reason = _passes_filter(
            history,
            val_auc_pr = metrics['val_auc_pr'],
            val_cost   = metrics['val_cost_per_veh'],
        )
 
        # 6. Store run regardless of outcome
        run = {
            'seed'          : seed,
            'history'       : history,
            'passed'        : passed_filter,
            'filter_reason' : reason,
            **metrics,
        }
 
        if passed_filter:
            passed.append(run)
            print(f"  ✓ PASS")
        else:
            rejected.append(run)
            print(f"  ✗ REJECT — {reason}")
 
        print(f"  Val AUC-PR:    {metrics['val_auc_pr']:.4f}")
        print(f"  Val cost/veh:  {metrics['val_cost_per_veh']:.4f}")
        print(f"  Test cost/veh: {metrics['test_cost_per_veh']:.4f}")
        print(f"  Threshold:     {metrics['optimal_threshold']:.2f}")
 
    # ── Aggregation over passing runs only ────────────────────────────────────
    AGGREGATE_METRICS = [
        'val_auc_pr', 'val_cost_per_veh',
        'test_auc_pr', 'test_auc_roc',
        'test_recall', 'test_precision', 'test_f1',
        'test_cost_per_veh', 'optimal_threshold',
    ]
 
    summary = {}
    if passed:
        for metric in AGGREGATE_METRICS:
            vals = np.array([r[metric] for r in passed])
            summary[metric] = {
                'mean': float(vals.mean()),
                'std' : float(vals.std()),
                'n'   : len(vals),
            }
 
    # ── Final report ──────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  {model_name} VERIFICATION COMPLETE")
    print(f"  {len(passed)}/{len(seeds)} runs passed the quality filter")
    print(f"{'='*55}")
 
    if summary:
        print(f"\n  {'Metric':<25} {'Mean':>10} {'Std':>10}")
        print(f"  {'-'*45}")
        for metric in AGGREGATE_METRICS:
            m = summary[metric]['mean']
            s = summary[metric]['std']
            print(f"  {metric:<25} {m:>10.4f} {s:>10.4f}")
 
    if len(passed) < 3:
        print(
            f"\n  ⚠ WARNING: Only {len(passed)} valid runs. "
            "Consider adding more seeds. High rejection rate "
            "indicates architectural instability — report this."
        )
 
    return {
        'passed'  : passed,
        'rejected': rejected,
        'summary' : summary,
    }
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Plotting functions
# ─────────────────────────────────────────────────────────────────────────────
 
def plot_training_curves(results: dict, model_name: str):
    """
    Plot 1: Individual training curves per seed.
    Each seed gets a unique colour consistent across all 3 subplots.
    Passed runs = solid lines. Rejected runs = dashed lines with star markers.
    """
    passed   = results['passed']
    rejected = results['rejected']
    all_runs = passed + rejected
 
    # Assign a unique colour to every seed
    palette    = plt.cm.tab10.colors + plt.cm.Set2.colors  # 18 distinct colours
    all_seeds  = sorted(set(r['seed'] for r in all_runs))
    seed_color = {seed: palette[i % len(palette)] for i, seed in enumerate(all_seeds)}
 
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics   = ['train_loss', 'val_loss', 'val_auc_pr']
    titles    = ['Train Loss', 'Val Loss', 'Val AUC-PR']
 
    for ax, metric, title in zip(axes, metrics, titles):
        # Rejected — dashed with star markers, faded
        for r in rejected:
            c = seed_color[r['seed']]
            ax.plot(r['history'][metric], color=c, alpha=0.5,
                    linewidth=1.2, linestyle='--',
                    marker='*', markevery=max(1, len(r['history'][metric])//8),
                    markersize=7, label=f"seed={r['seed']} ✗")
 
        # Passed — solid lines, full opacity
        for r in passed:
            c = seed_color[r['seed']]
            ax.plot(r['history'][metric], color=c, alpha=0.85,
                    linewidth=1.2, label=f"seed={r['seed']} ✓")
 
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.set_title(f'{model_name} — {title} per seed')
        ax.grid(True, alpha=0.3)
 
    # Build clean legend from seed colours, one entry per seed
    legend_handles = []
    for seed in all_seeds:
        c    = seed_color[seed]
        run  = next(r for r in all_runs if r['seed'] == seed)
        mark = '✓' if run['passed'] else '✗'
        ls   = '-' if run['passed'] else '--'
        legend_handles.append(
            plt.Line2D([0], [0], color=c, linewidth=1.5,
                       linestyle=ls, label=f"seed={seed} {mark}")
        )
 
    fig.legend(handles=legend_handles, loc='upper right',
               fontsize=8, ncol=2, title='Seeds')
    plt.suptitle(
        f'{model_name} — Training Curves per Seed  '
        f'(solid=passed, dashed★=rejected)',
        fontsize=13
    )
    plt.tight_layout()
    plt.show()
 
 
def plot_mean_training_curve(results: dict, model_name: str):
    """
    Plot 2: Mean training curve with ±1 std shaded band.
    Truncated to shortest passing run length so every epoch
    has the same N contributing runs.
    """
    passed = results['passed']
    if not passed:
        print(f"No passing runs for {model_name} — cannot plot mean curve.")
        return
 
    color   = 'steelblue' if model_name.upper() == 'LSTM' else 'darkorange'
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics   = ['train_loss', 'val_loss', 'val_auc_pr']
    titles    = ['Train Loss', 'Val Loss', 'Val AUC-PR']
 
    for ax, metric, title in zip(axes, metrics, titles):
        # Truncate to shortest run
        min_len = min(len(r['history'][metric]) for r in passed)
        matrix  = np.array([r['history'][metric][:min_len] for r in passed])
 
        mean_curve = matrix.mean(axis=0)
        std_curve  = matrix.std(axis=0)
        epochs     = np.arange(min_len)
 
        ax.plot(epochs, mean_curve, color=color, linewidth=2, label='Mean')
        ax.fill_between(epochs,
                        mean_curve - std_curve,
                        mean_curve + std_curve,
                        alpha=0.2, color=color, label='±1 std')
 
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.set_title(f'{model_name} — Mean {title}\n(n={len(passed)} passing runs, truncated to {min_len} epochs)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
 
    plt.suptitle(f'{model_name} — Mean Training Curve ± Std', fontsize=13)
    plt.tight_layout()
    plt.show()
 
 
def plot_per_seed_scatter(results: dict, model_name: str):
    """Per-seed scatter for test cost/vehicle and test AUC-PR."""
    color  = 'steelblue' if model_name.upper() == 'LSTM' else 'darkorange'
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    metrics   = ['test_cost_per_veh', 'test_auc_pr']
    titles    = ['Test Cost / Vehicle (lower is better)', 'Test AUC-PR (higher is better)']
 
    for ax, metric, title in zip(axes, metrics, titles):
        for r in results['passed']:
            ax.scatter(r['seed'], r[metric], color=color, s=80, zorder=3, label='passed')
        for r in results['rejected']:
            ax.scatter(r['seed'], r[metric], color='red', s=120,
                       marker='x', zorder=4, linewidths=2.5, label='rejected')
        if results['summary'] and metric in results['summary']:
            mean_val = results['summary'][metric]['mean']
            ax.axhline(mean_val, color=color, linestyle='--',
                       linewidth=1.5, label=f'Mean={mean_val:.3f}')
        if metric == 'test_cost_per_veh':
            ax.axhline(8.62, color='red', linestyle=':', linewidth=1, label='Predict-all-0 (8.62)')
        ax.set_xlabel('Seed')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        seen = {}
        for h, l in zip(handles, labels):
            if l not in seen:
                seen[l] = h
        ax.legend(seen.values(), seen.keys(), fontsize=8)
 
    plt.suptitle(f'{model_name} — Per-Seed Results (× = rejected)', fontsize=13)
    plt.tight_layout()
    plt.show()
 
 
def plot_boxplot(results: dict, model_name: str):
    """Box plot of test cost/vehicle and test AUC-PR across passing runs."""
    color  = 'steelblue' if model_name.upper() == 'LSTM' else 'darkorange'
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    metrics   = ['test_cost_per_veh', 'test_auc_pr']
    titles    = ['Test Cost / Vehicle', 'Test AUC-PR']
 
    for ax, metric, title in zip(axes, metrics, titles):
        vals = [r[metric] for r in results['passed']]
        if not vals:
            ax.set_title(f'{title} — no passing runs')
            continue
        bp = ax.boxplot([vals], patch_artist=True,
                        labels=[f'{model_name}\n(n={len(vals)})'],
                        medianprops=dict(color='black', linewidth=2))
        bp['boxes'][0].set_facecolor(color)
        bp['boxes'][0].set_alpha(0.7)
        jitter = np.random.uniform(-0.05, 0.05, size=len(vals))
        ax.scatter([1 + j for j in jitter], vals, color=color, s=40, zorder=3, alpha=0.8)
        if metric == 'test_cost_per_veh':
            ax.axhline(8.62, color='red', linestyle=':', linewidth=1, label='Predict-all-0 (8.62)')
            ax.legend(fontsize=9)
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3)
 
    plt.suptitle(f'{model_name} — Result Distribution Across Seeds', fontsize=13)
    plt.tight_layout()
    plt.show()
 
 
def plot_all(results: dict, model_name: str):
    """
    Plots all 4 plots for a single model.
    Usage: plot_all(lstm_results, 'LSTM') or plot_all(tcn_results, 'TCN')
    """
    plot_training_curves(results, model_name)
    plot_mean_training_curve(results, model_name)
    plot_per_seed_scatter(results, model_name)
    plot_boxplot(results, model_name)