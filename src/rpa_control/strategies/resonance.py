import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class ResonanceStrategy:
    """
    Parametric resonance strategy for an RPAModel.

    For each RPA parameter two variants are created:
        '2x'   — parameter set to 2 × baseline
        '0.5x' — parameter set to 0.5 × baseline

    At each RK4 step the spring is chosen from the current O and dO/dt:
        hard  when O is moving toward  O*   (accelerate return)
        soft  when O is moving away from O* (reduce overshoot)

    Attributes
    ----------
    variants  : {param: {'2x': overrides_dict, '0.5x': overrides_dict}}
    O_star    : float — steady-state output value at base parameters
    results   : {param: {'2x': result_dict, '0.5x': result_dict}}
    """

    def __init__(self, model):
        self.model    = model
        self.baseline = dict(model.params_base)
        self._oidx    = model.output_idx

        self.variants = {
            param: {
                '2x':   {param: self.baseline[param] * 2.0},
                '0.5x': {param: self.baseline[param] * 0.5},
            }
            for param in model.rpa_params
        }

        self.O_star = float(self._eval_ss()[self._oidx])
        self.results = {}

    def _eval_ss(self, overrides=None):
        p = dict(self.baseline)
        if overrides:
            p.update(overrides)
        subs = {sym: p[name]
                for name, sym in zip(self.model._param_names, self.model._sym_params)}
        return np.array([float(self.model._ss_dict[s].subs(subs))
                         for s in self.model._sym_state])

    def _get_f(self, param, key):
        return self.model.get_f(self.variants[param][key])

    def simulate(self, param, hard_key='2x', t_end=500, perturb_factor=1.3,
                 dt=0.005, min_dwell=0.0):
        """
        Fixed-step RK4 simulation with adaptive spring switching.

        At every step the spring is chosen based on O vs O* and sign of dO/dt.
        min_dwell sets a minimum time (in model time units) that must elapse
        between consecutive switches, preventing rapid chattering.
        """
        soft_key   = '0.5x' if hard_key == '2x' else '2x'
        oidx       = self._oidx
        O_star     = self.O_star
        hard_val   = self.baseline[param] * (2.0 if hard_key == '2x' else 0.5)
        soft_val   = self.baseline[param] * (0.5 if hard_key == '2x' else 2.0)

        f_hard = self._get_f(param, hard_key)
        f_soft = self._get_f(param, soft_key)

        y0         = self._eval_ss(self.variants[param][hard_key]).copy()
        y0[oidx]  *= perturb_factor

        n_steps      = int(round(t_end / dt))
        dwell_steps  = int(round(min_dwell / dt))
        t_arr        = np.linspace(0, t_end, n_steps + 1)
        y_arr        = np.zeros((len(y0), n_steps + 1))
        y_arr[:, 0]  = y0
        ctrl         = np.zeros(n_steps + 1)

        switches          = []
        prev_key          = None
        use_hard          = True
        steps_since_switch = dwell_steps

        for i in range(n_steps):
            y_i    = y_arr[:, i]
            O_i    = y_i[oidx]
            dOdt_i = f_hard(t_arr[i], y_i)[oidx]

            if O_i > O_star:
                desired_hard = dOdt_i < 0
            else:
                desired_hard = dOdt_i > 0

            if desired_hard != use_hard and steps_since_switch >= dwell_steps:
                use_hard           = desired_hard
                steps_since_switch = 0

            key     = hard_key if use_hard else soft_key
            ctrl[i] = hard_val if use_hard else soft_val

            if key != prev_key:
                if prev_key is not None:
                    switches.append((t_arr[i], prev_key, key))
                prev_key = key

            f_cur = f_hard if use_hard else f_soft
            t_i   = t_arr[i]
            k1 = np.array(f_cur(t_i,          y_i))
            k2 = np.array(f_cur(t_i + 0.5*dt, y_i + 0.5*dt*k1))
            k3 = np.array(f_cur(t_i + 0.5*dt, y_i + 0.5*dt*k2))
            k4 = np.array(f_cur(t_i + dt,      y_i +      dt*k3))
            y_arr[:, i+1] = y_i + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

            steps_since_switch += 1

        ctrl[-1] = ctrl[-2]

        self.results.setdefault(param, {})[hard_key] = {
            't':        t_arr,
            'y':        y_arr,
            'ctrl':     ctrl,
            'switches': switches,
            'O_star':   O_star,
            'hard_key': hard_key,
            'soft_key': soft_key,
            'hard_val': hard_val,
            'soft_val': soft_val,
        }
        return t_arr, y_arr

    def simulate_all(self, **kwargs):
        """Run both variants for every RPA parameter."""
        for param in self.model.rpa_params:
            for hard_key in ('2x', '0.5x'):
                print(f"  param={param}  hard={hard_key} ...")
                self.simulate(param, hard_key=hard_key, **kwargs)
        print("Done.")


def run_resonance_demo(model, t_end=10, perturb_factor=1.3, dt=0.005, min_dwell=0.0):
    """
    Build a ResonanceStrategy for `model`, simulate both variants (hard='2x'
    and hard='0.5x') for every RPA parameter, and plot O(t) with spring shading.

    Layout: n_params rows × 2 columns.
      Left column  — hard = 2×, soft = 0.5×
      Right column — hard = 0.5×, soft = 2×

    Returns the ResonanceStrategy instance.
    """
    rs = ResonanceStrategy(model)
    print(f"Model {model.name}  |  O* = {rs.O_star:.4f}  |  "
          f"RPA params: {model.rpa_params}\n")
    rs.simulate_all(t_end=t_end, perturb_factor=perturb_factor, dt=dt,
                    min_dwell=min_dwell)

    n = len(model.rpa_params)
    fig, axes = plt.subplots(n, 2, figsize=(14, 3.5 * n), sharex=False)
    if n == 1:
        axes = axes.reshape(1, 2)

    for row, param in enumerate(model.rpa_params):
        for col, hard_key in enumerate(('2x', '0.5x')):
            ax       = axes[row, col]
            res      = rs.results[param][hard_key]
            t        = res['t']
            y        = res['y']
            ctrl     = res['ctrl']
            oidx     = model.output_idx
            hard_val = res['hard_val']
            soft_val = res['soft_val']

            is_hard = np.isclose(ctrl, hard_val)
            changes = np.where(np.diff(is_hard.astype(int)) != 0)[0]
            bounds  = np.concatenate([[0], changes + 1, [len(t) - 1]])
            for j in range(len(bounds) - 1):
                i0, i1 = bounds[j], bounds[j + 1]
                color  = '#e8c4a0' if is_hard[i0] else '#c4d8e8'
                ax.axvspan(t[i0], t[i1], alpha=0.25, color=color, lw=0)

            ax.plot(t, y[oidx], color='steelblue', lw=1.5)
            ax.axhline(res['O_star'], color='k', ls='--', lw=1)

            ax.legend(handles=[
                plt.Line2D([0], [0], color='steelblue', lw=1.5, label='O(t)'),
                plt.Line2D([0], [0], color='k', ls='--', lw=1,
                           label=f"O* = {res['O_star']:.3f}"),
                mpatches.Patch(facecolor='#e8c4a0', alpha=0.5,
                               label=f'toward O*: {param} = {hard_val:.3g}'),
                mpatches.Patch(facecolor='#c4d8e8', alpha=0.5,
                               label=f'away from O*: {param} = {soft_val:.3g}'),
            ], fontsize=7.5)

            ax.set_title(
                f"param = {param}  |  hard = {hard_key} ({hard_val:.3g}),  "
                f"soft = {'0.5x' if hard_key == '2x' else '2x'} ({soft_val:.3g})  "
                f"|  {len(res['switches'])} switches",
                fontsize=9)
            ax.set_xlabel('time')
            ax.set_ylabel('O')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    fig.suptitle(f'Resonance strategy — model {model.name}', fontsize=11, y=1.01)
    plt.tight_layout()
    plt.show()
    return rs
