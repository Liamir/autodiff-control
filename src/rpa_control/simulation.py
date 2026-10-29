import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def simulate_rpa_on_off(model, parameter_name, on_value, off_value, t_on, t_off,
                        y0=None, rtol=1e-8, atol=1e-10, plot=True):
    """
    Simulate one on→off treatment cycle for an RPAModel or ABCModel.

    If y0 is None the analytical SS at off_value is used (RPAModel only).
    Pass y0 explicitly for models without an analytic SS (e.g. ABCModel).

    Set plot=False to skip the default per-variable figure (e.g. when the
    caller wants to build its own custom plot from the returned solutions).

    Returns
    -------
    sol_on, sol_off : scipy OdeResult for the two phases
    """
    if y0 is None:
        p = dict(model.params_base)
        p[parameter_name] = off_value
        subs = {sym: p[name]
                for name, sym in zip(model._param_names, model._sym_params)}
        y0 = np.array([float(model._ss_dict[s].subs(subs)) for s in model._sym_state])
        if getattr(model, '_extra_vars', None):
            y0[len(model.state_vars) - len(model._extra_vars):] = 1.0
    else:
        y0 = np.asarray(y0, dtype=float)

    kw = dict(method='Radau', dense_output=True, rtol=rtol, atol=atol)
    sol_on  = solve_ivp(model.get_f({parameter_name: on_value}),
                        (0, t_on), y0, **kw)
    sol_off = solve_ivp(model.get_f({parameter_name: off_value}),
                        (0, t_off), sol_on.y[:, -1], **kw)

    if not plot:
        return sol_on, sol_off

    t_all = np.concatenate([sol_on.t, sol_off.t + t_on])
    y_all = np.concatenate([sol_on.y, sol_off.y], axis=1)

    ON_COLOR = 'lightgray'
    n_vars   = len(model.state_vars)
    fig, axes = plt.subplots(n_vars, 1, figsize=(8, 2.0 * n_vars), sharex=True,
                              gridspec_kw={'hspace': 0.08})
    if n_vars == 1:
        axes = [axes]

    for i, (ax, vname) in enumerate(zip(axes, model.state_vars)):
        ax.axvspan(0, t_on, color=ON_COLOR, alpha=0.4, lw=0)
        ax.axvline(t_on, color='k', ls=':', lw=0.9)
        ax.plot(t_all, y_all[i], color=f'C{i % 10}', lw=1.5)
        ax.axhline(y0[i], color=f'C{i % 10}', ls=':', lw=0.8, alpha=0.6,
                   label=f'SS = {y0[i]:.3f}')
        ax.set_ylabel(vname, fontsize=9)
        ax.legend(fontsize=7.5, frameon=False, loc='upper right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[-1].set_xlabel('time')
    axes[0].set_title(
        f"Model {model.name}  |  {parameter_name}: on = {on_value:.3g},  "
        f"off = {off_value:.3g}  |  t_on = {t_on},  t_off = {t_off}",
        fontsize=9)
    plt.tight_layout()
    plt.show()
    return sol_on, sol_off


def simulate_rpa_intermittent(model, parameter_name, on_value, off_value,
                               t_on, t_off, n_cycles,
                               y0=None, rtol=1e-8, atol=1e-10, plot=True):
    """
    Simulate intermittent (periodic) treatment for an RPAModel or ABCModel.

    If y0 is None the analytical SS at off_value is used (RPAModel only).
    Pass y0 explicitly for models without an analytic SS (e.g. ABCModel).

    Set plot=False to skip the default per-variable figure (e.g. when the
    caller wants to build its own custom plot from the returned trajectory).

    Returns
    -------
    t_all : concatenated time array
    y_all : (n_vars, n_points) concatenated trajectory
    """
    if y0 is None:
        p = dict(model.params_base)
        p[parameter_name] = off_value
        subs = {sym: p[name]
                for name, sym in zip(model._param_names, model._sym_params)}
        y0 = np.array([float(model._ss_dict[s].subs(subs)) for s in model._sym_state])
        if getattr(model, '_extra_vars', None):
            y0[len(model.state_vars) - len(model._extra_vars):] = 1.0
    else:
        y0 = np.asarray(y0, dtype=float)

    f_on  = model.get_f({parameter_name: on_value})
    f_off = model.get_f({parameter_name: off_value})
    kw    = dict(method='Radau', dense_output=True, rtol=rtol, atol=atol)

    t_segs, y_segs = [], []
    t_offset = 0.0
    y_cur    = y0.copy()

    for _ in range(n_cycles):
        for f_cur, duration in [(f_on, t_on), (f_off, t_off)]:
            sol = solve_ivp(f_cur, (0, duration), y_cur, **kw)
            t_segs.append(sol.t + t_offset)
            y_segs.append(sol.y)
            t_offset += duration
            y_cur     = sol.y[:, -1]

    t_all = np.concatenate(t_segs)
    y_all = np.concatenate(y_segs, axis=1)

    if not plot:
        return t_all, y_all

    ON_COLOR = 'lightgray'
    on_ranges = [(c * (t_on + t_off), c * (t_on + t_off) + t_on)
                 for c in range(n_cycles)]

    n_vars = len(model.state_vars)
    fig, axes = plt.subplots(n_vars, 1, figsize=(8, 2.0 * n_vars), sharex=True,
                              gridspec_kw={'hspace': 0.08})
    if n_vars == 1:
        axes = [axes]

    for i, (ax, vname) in enumerate(zip(axes, model.state_vars)):
        for t_start, t_end in on_ranges:
            ax.axvspan(t_start, t_end, color=ON_COLOR, alpha=0.4, lw=0)
        ax.plot(t_all, y_all[i], color=f'C{i % 10}', lw=1.5)
        ax.axhline(y0[i], color=f'C{i % 10}', ls=':', lw=0.8, alpha=0.6,
                   label=f'SS = {y0[i]:.3f}')
        ax.set_ylabel(vname, fontsize=9)
        ax.legend(fontsize=7.5, frameon=False, loc='upper right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[-1].set_xlabel('time')
    axes[0].set_title(
        f"Model {model.name}  |  {parameter_name}: on = {on_value:.3g},  "
        f"off = {off_value:.3g}  |  t_on = {t_on},  t_off = {t_off},  "
        f"n_cycles = {n_cycles}",
        fontsize=9)
    plt.tight_layout()
    plt.show()
    return t_all, y_all


def simulate_threshold_driven(model, parameter_name, on_value, off_value,
                              burden_fn, off_frac, on_frac, T_horizon, y0,
                              rtol=1e-8, atol=1e-10, max_step=0.5):
    """
    Simulate burden-threshold-driven (PSA-style) adaptive treatment: start
    with treatment ON; switch OFF once burden_fn(y) drops to off_frac times
    its initial value; switch back ON once it regrows to on_frac times its
    initial value (repeat until T_horizon). Mimics clinical adaptive-therapy
    protocols that react to a measurable tumor-burden proxy (e.g. PSA)
    rather than to the underlying fast homeostatic dynamics.

    burden_fn(y) -> scalar burden metric, e.g. lambda y: y[2] + y[3] for a
    two-clone model's total tumor size.

    Returns
    -------
    t_all, y_all : concatenated trajectory (as in simulate_rpa_intermittent)
    on_ranges    : list of (t_start, t_end) treatment-on intervals actually
                   realised (durations are determined by the dynamics, not
                   fixed in advance)
    """
    y0 = np.asarray(y0, dtype=float)
    baseline      = burden_fn(y0)
    off_threshold = off_frac * baseline
    on_threshold  = on_frac  * baseline

    f_on  = model.get_f({parameter_name: on_value})
    f_off = model.get_f({parameter_name: off_value})

    def event_off(t, y):
        return burden_fn(y) - off_threshold
    event_off.terminal  = True
    event_off.direction = -1

    def event_on(t, y):
        return burden_fn(y) - on_threshold
    event_on.terminal  = True
    event_on.direction = 1

    t_cur, y_cur, treatment_on = 0.0, y0.copy(), True
    t_segs, y_segs, on_ranges  = [], [], []

    while t_cur < T_horizon - 1e-9:
        f_cur = f_on if treatment_on else f_off
        event = event_off if treatment_on else event_on
        sol = solve_ivp(f_cur, (0, T_horizon - t_cur), y_cur, events=event,
                        max_step=max_step, rtol=rtol, atol=atol)
        assert sol.success, f"integration failed: {sol.message}"
        seg_t = sol.t + t_cur
        t_segs.append(seg_t)
        y_segs.append(sol.y)
        seg_end = seg_t[-1]
        if treatment_on:
            on_ranges.append((t_cur, seg_end))
        t_cur, y_cur = seg_end, sol.y[:, -1]
        if sol.t_events[0].size == 0:
            break   # reached T_horizon without another threshold crossing
        treatment_on = not treatment_on

    t_all = np.concatenate(t_segs)
    y_all = np.concatenate(y_segs, axis=1)
    return t_all, y_all, on_ranges


def validate_rpa_params(model, scale=2.0, t_end=500, ss_tol=1e-2,
                        rtol=1e-10, atol=1e-12):
    """
    For every declared RPA parameter, run the ODE at the base value and at
    scale*base value. Both runs start from y0 = 0.5 * ones and integrate to t_end.

    Returns
    -------
    dict mapping param_name -> {base_val, test_val, O_base, O_scaled,
                                 rel_diff, passed}
    """
    params_to_test = model.rpa_params

    y0  = np.ones(len(model.state_vars)) * 0.5
    kw  = dict(method='Radau', rtol=rtol, atol=atol, dense_output=False)

    sol_base = solve_ivp(model.get_f(), (0, t_end), y0, **kw)
    assert sol_base.success, f"Base integration failed for model {model.name}"
    O_base = sol_base.y[model.output_idx, -1]

    results = {}
    for param in params_to_test:
        base_val = model.params_base[param]
        test_val = base_val * scale
        sol = solve_ivp(model.get_f({param: test_val}), (0, t_end), y0, **kw)
        assert sol.success, f"Integration failed for model {model.name}, {param}={test_val}"
        O_test   = sol.y[model.output_idx, -1]
        rel_diff = abs(O_test - O_base) / abs(O_base)
        results[param] = dict(
            base_val = base_val,
            test_val = test_val,
            O_base   = O_base,
            O_scaled = O_test,
            rel_diff = rel_diff,
            passed   = rel_diff < ss_tol,
        )
    return results


def print_validation_results(model, results):
    n_pass  = sum(r['passed'] for r in results.values())
    n_total = len(results)
    tag     = 'PASS' if n_pass == n_total else 'FAIL'
    print(f"[{tag}] Case {model.name}  ({n_pass}/{n_total} params)")
    for param, r in results.items():
        mark = '✓' if r['passed'] else '✗'
        print(f"  {mark} {param:5s}  base={r['base_val']:.3f} → {r['test_val']:.3f}"
              f"  |  O*= {r['O_base']:.5f}  vs  {r['O_scaled']:.5f}"
              f"  (rel Δ = {r['rel_diff']:.1e})")


def classify_rpa_params(model, T_step=5.0, dt=0.02, tol=0.005):
    """
    Step each RPA parameter to 2× baseline from the base SS, classify the
    output response as pos (O rises above O*), neg (O drops below O*), or none.

    Sets model.rpa_params_pos, model.rpa_params_neg, model.rpa_params_none
    and returns (pos_params, neg_params, none_params).
    """
    subs = {sym: model.params_base[name]
            for name, sym in zip(model._param_names, model._sym_params)}
    ss_base = np.array([float(model._ss_dict[s].subs(subs)) for s in model._sym_state])
    if getattr(model, '_extra_vars', None):
        n_base = len(model.state_vars) - len(model._extra_vars)
        ss_base[n_base:] = 1.0

    O_star = ss_base[model.output_idx]
    t_eval = np.arange(0, T_step, dt)

    pos_params, neg_params, none_params = [], [], []
    for param in model.rpa_params:
        f_step = model.get_f({param: model.params_base[param] * 2.0})
        sol = solve_ivp(f_step, (0, T_step), ss_base, t_eval=t_eval,
                        method='Radau', rtol=1e-8, atol=1e-10)
        O_mean = sol.y[model.output_idx].mean()

        if O_mean > O_star * (1 + tol):
            pos_params.append(param)
        elif O_mean < O_star * (1 - tol):
            neg_params.append(param)
        else:
            none_params.append(param)

    model.rpa_params_pos  = pos_params
    model.rpa_params_neg  = neg_params
    model.rpa_params_none = none_params
    return pos_params, neg_params, none_params


def stepped_trajectory(model, param, values, durations, dt=0.2, y0=None):
    """
    Simulate a continuous trajectory with piecewise-constant parameter steps.

    Parameters
    ----------
    param     : parameter name to step (must be in model.params_base)
    values    : list of parameter values, one per step
    durations : list of step durations (same length as values)
    dt        : time-step for output evaluation (default 0.2)
    y0        : initial condition; defaults to SS at {param: values[0]}
    """
    if y0 is None:
        y0 = eval_ss(model, {param: values[0]})
        if getattr(model, '_extra_vars', None):
            n_base = len(model.state_vars) - len(model._extra_vars)
            y0[n_base:] = 1.0
    y_cur, t_offset = y0.copy(), 0.0
    t_all, y_all = [], []
    step_starts = []
    for val, dur in zip(values, durations):
        step_starts.append(t_offset)
        t_eval = np.arange(0, dur, dt)
        sol = solve_ivp(model.get_f({param: val}), (0, dur), y_cur,
                        t_eval=t_eval, method='Radau', rtol=1e-8, atol=1e-10)
        t_all.append(sol.t + t_offset)
        y_all.append(sol.y)
        y_cur     = sol.y[:, -1]
        t_offset += dur

    t_all = np.concatenate(t_all)
    y_all = np.concatenate(y_all, axis=1)

    # ── plot ──────────────────────────────────────────────────────────────────
    n_vars = len(model.state_vars)
    fig, axes = plt.subplots(n_vars + 1, 1, figsize=(9, (n_vars + 1) * 1.2),
                             sharex=True, gridspec_kw={'hspace': 0.08})

    # parameter staircase
    t_stairs = np.repeat([0] + [s for s in step_starts[1:]] + [t_offset], 2)[1:-1]
    v_stairs = np.repeat(values, 2)
    axes[0].plot(t_stairs, v_stairs, color='k', lw=1.5)
    axes[0].set_ylabel(param, fontsize=8)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    # state variables
    for i, (ax, vname) in enumerate(zip(axes[1:], model.state_vars)):
        # print(f"Plotting {vname} with color C{i % 10}")
        ax.plot(t_all, y_all[i], color=f'C{i % 10}', lw=1.5)
        for t_s in step_starts[1:]:
            ax.axvline(t_s, color='grey', ls=':', lw=0.8)
        ax.set_ylabel(vname, fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if vname == 'Cancer':
            # make the y-axis log-scale
            ax.set_yscale('log')

    axes[-1].set_xlabel('time')
    axes[0].set_title(f'Model {model.name}  |  stepped {param}', fontsize=9)
    plt.tight_layout()
    plt.show()

    return t_all, y_all


def eval_ss(model, overrides=None):
    """Evaluate the symbolic steady state with optional parameter overrides."""
    p = dict(model.params_base)
    if overrides:
        p.update(overrides)
    subs = {sym: p[name]
            for name, sym in zip(model._param_names, model._sym_params)}
    return np.array([float(model._ss_dict[s].subs(subs)) for s in model._sym_state])
