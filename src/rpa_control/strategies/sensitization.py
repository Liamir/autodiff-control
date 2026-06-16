import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class SensitizationStrategy:
    """
    Two-stage sensitization + treatment strategy for an RPAModel.

    Stage 1 — sensitization: linearly ramp the parameter from its baseline
              value to a primed value over ramp_duration time units.
    Stage 2 — treatment: abruptly switch the parameter to the treatment value
              and simulate for t_post time units.

    A control trajectory is also computed: the system holds at the baseline SS
    for the entire ramp phase (no sensitization), then receives the same
    abrupt treatment at the same time.

    Two variants per RPA parameter:
        'up_down': baseline → 2× (slow ramp),  then → 0.5× (abrupt)
        'down_up': baseline → 0.5× (slow ramp), then → 2× (abrupt)

    Results dict keys
    -----------------
    t, y          : time array and sensitized state trajectory
    y_ctrl        : control trajectory (no ramp, direct treatment)
    param_val     : parameter value at every time point (sensitized path)
    base_val, primed_val, treatment_val, ramp_duration, O_star
    """

    def __init__(self, model):
        self.model    = model
        self.baseline = dict(model.params_base)
        self._oidx    = model.output_idx
        self.O_star   = float(self._eval_ss()[self._oidx])
        self.results  = {}

    def _eval_ss(self, overrides=None):
        p = dict(self.baseline)
        if overrides:
            p.update(overrides)
        subs = {sym: p[name]
                for name, sym in zip(self.model._param_names, self.model._sym_params)}
        return np.array([float(self.model._ss_dict[s].subs(subs))
                         for s in self.model._sym_state])

    def simulate(self, param, variant='up_down', ramp_duration=10.0,
                 t_post=10.0, dt=0.005, fast_prep=False,
                 start_from_primed_ss=False):
        """
        Simulate one (param, variant) pair plus a no-ramp control.

        variant : 'up_down' (ramp 1×→2×, treat 0.5×) or
                  'down_up' (ramp 1×→0.5×, treat 2×)
        """
        base_val = self.baseline[param]
        _pos  = getattr(self.model, 'rpa_params_pos',  [])
        _neg  = getattr(self.model, 'rpa_params_neg',  [])
        if param in _pos:
            if variant == 'up_down':
                primed_val, treatment_val = base_val * 2.0, base_val * 0.5
            else:
                primed_val, treatment_val = base_val * 0.5, base_val * 2.0
        elif param in _neg:
            if variant == 'up_down':
                primed_val, treatment_val = base_val * 0.5, base_val * 2.0
            else:
                primed_val, treatment_val = base_val * 2.0, base_val * 0.5
        else:
            if variant == 'up_down':
                primed_val, treatment_val = base_val * 2.0, base_val * 0.5
            else:
                primed_val, treatment_val = base_val * 0.5, base_val * 2.0

        n_ramp  = int(round(ramp_duration / dt))
        n_post  = int(round(t_post / dt))
        n_total = n_ramp + n_post

        t_arr  = np.linspace(0, ramp_duration + t_post, n_total + 1)
        y_arr  = np.zeros((len(self.model.state_vars), n_total + 1))
        y_ctrl = np.zeros_like(y_arr)
        p_arr  = np.zeros(n_total + 1)

        ss_base = self._eval_ss()
        if getattr(self.model, '_extra_vars', None):
            n_base = len(self.model.state_vars) - len(self.model._extra_vars)
            ss_base[n_base:] = 1.0

        if start_from_primed_ss:
            _override = {param: primed_val}
            ss_sens = self._eval_ss(_override)
            if getattr(self.model, '_extra_vars', None):
                ss_sens[n_base:] = 1.0
        else:
            ss_sens = ss_base

        y_arr[:, 0]  = ss_sens
        y_ctrl[:, 0] = ss_base
        p_arr[0]     = base_val

        ramp_vals = np.linspace(base_val, primed_val, n_ramp + 1)
        f_treat   = self.model.get_f({param: treatment_val})
        f_base    = self.model.get_f({param: base_val})
        f_primed  = self.model.get_f({param: primed_val})

        for i in range(n_ramp):
            if fast_prep:
                p_i   = primed_val
                f_cur = f_primed
            else:
                p_i   = ramp_vals[i]
                f_cur = self.model.get_f({param: p_i})
            p_arr[i] = p_i
            t_i      = t_arr[i]

            y_i  = y_arr[:, i]
            k1 = np.array(f_cur(t_i,           y_i))
            k2 = np.array(f_cur(t_i + 0.5*dt,  y_i + 0.5*dt*k1))
            k3 = np.array(f_cur(t_i + 0.5*dt,  y_i + 0.5*dt*k2))
            k4 = np.array(f_cur(t_i + dt,       y_i +      dt*k3))
            y_arr[:, i+1] = y_i + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

            yc_i = y_ctrl[:, i]
            c1 = np.array(f_base(t_i,           yc_i))
            c2 = np.array(f_base(t_i + 0.5*dt,  yc_i + 0.5*dt*c1))
            c3 = np.array(f_base(t_i + 0.5*dt,  yc_i + 0.5*dt*c2))
            c4 = np.array(f_base(t_i + dt,       yc_i +      dt*c3))
            y_ctrl[:, i+1] = yc_i + (dt / 6.0) * (c1 + 2*c2 + 2*c3 + c4)

        for i in range(n_ramp, n_total):
            p_arr[i] = treatment_val
            for y_cur_arr in (y_arr, y_ctrl):
                y_i, t_i = y_cur_arr[:, i], t_arr[i]
                k1 = np.array(f_treat(t_i,           y_i))
                k2 = np.array(f_treat(t_i + 0.5*dt,  y_i + 0.5*dt*k1))
                k3 = np.array(f_treat(t_i + 0.5*dt,  y_i + 0.5*dt*k2))
                k4 = np.array(f_treat(t_i + dt,       y_i +      dt*k3))
                y_cur_arr[:, i+1] = y_i + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        p_arr[n_ramp] = treatment_val
        p_arr[-1]     = treatment_val

        self.results.setdefault(param, {})[variant] = {
            't':             t_arr,
            'y':             y_arr,
            'y_ctrl':        y_ctrl,
            'param_val':     p_arr,
            'base_val':      base_val,
            'primed_val':    primed_val,
            'treatment_val': treatment_val,
            'ramp_duration': ramp_duration,
            'fast_prep':     fast_prep,
            'O_star':        self.O_star,
        }
        return t_arr, y_arr

    def simulate_all(self, variant='auto', start_from_primed_ss=False, **kwargs):
        kwargs['start_from_primed_ss'] = start_from_primed_ss
        v_use = 'up_down' if variant != 'down_up' else 'down_up'
        for param in self.model.rpa_params:
            print(f"  param={param}  variant={v_use} ...")
            self.simulate(param, variant=v_use, **kwargs)
        print("Done.")


def run_sensitization_demo(model, ramp_duration=10.0, t_post=30.0, dt=0.005,
                           cancer_threshold=2.0, fast_prep=False, variant='auto',
                           start_from_primed_ss=False):
    """
    Build a SensitizationStrategy for `model`, simulate both variants for
    every RPA parameter, and plot results.

    Each panel shows O(t), Cancer(t) (if present), and the parameter
    trajectory, overlaying sensitized vs no-ramp control.
    """
    ss = SensitizationStrategy(model)
    print(f"Model {model.name}  |  O* = {ss.O_star:.4f}  |  "
          f"RPA params: {model.rpa_params}\n")
    ss.simulate_all(ramp_duration=ramp_duration, t_post=t_post, dt=dt,
                    fast_prep=fast_prep, variant=variant,
                    start_from_primed_ss=start_from_primed_ss)

    RAMP_COLOR  = 'white'
    TREAT_COLOR = 'lightgray'

    has_cancer  = 'Cancer' in model.state_vars
    cancer_idx  = model.state_vars.index('Cancer') if has_cancer else None
    other_vars  = [(i, nm) for i, nm in enumerate(model.state_vars)
                   if i != model.output_idx and nm != 'Cancer']
    n_other     = len(other_vars)
    n_inner     = 1 + n_other + (1 if has_cancer else 0) + 1
    h_ratios    = [2] * (1 + n_other) + ([2] if has_cancer else []) + [1]

    v_use = 'up_down' if variant != 'down_up' else 'down_up'
    variants_to_show = (v_use,)
    n = len(model.rpa_params)
    fig = plt.figure(figsize=(8, 7 * n))
    outer = fig.add_gridspec(n, 1, hspace=0.45)

    for row, param in enumerate(model.rpa_params):
        for col, v in enumerate(variants_to_show):
            res  = ss.results[param][v]
            t    = res['t']
            y    = res['y']
            yc   = res['y_ctrl']
            p    = res['param_val']
            t_sw = res['ramp_duration']
            oidx = model.output_idx

            inner = outer[row, col].subgridspec(n_inner, 1, height_ratios=h_ratios,
                                                hspace=0.08)
            ax_O      = fig.add_subplot(inner[0])
            ax_others = [fig.add_subplot(inner[1 + i], sharex=ax_O)
                         for i in range(n_other)]
            ax_C      = fig.add_subplot(inner[1 + n_other], sharex=ax_O) if has_cancer else None
            ax_p      = fig.add_subplot(inner[-1], sharex=ax_O)

            active_axes = [ax_O] + ax_others + ([ax_C] if ax_C else []) + [ax_p]
            for ax in active_axes:
                ax.axvspan(t[0], t_sw,  color=RAMP_COLOR,  alpha=0.4, lw=0)
                ax.axvspan(t_sw, t[-1], color=TREAT_COLOR, alpha=0.4, lw=0)
                ax.axvline(t_sw, color='k', ls=':', lw=0.9, alpha=0.6)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

            _p = getattr(model, 'rpa_params_pos', None)
            _n = getattr(model, 'rpa_params_neg', None)
            if _p is not None:
                is_correct = param in (_p or []) or param in (_n or [])
            else:
                is_correct = True
            mark = '  ★' if is_correct else '  (none)'

            ax_O.plot(t, y[oidx],  color='steelblue', lw=1.8)
            ax_O.plot(t, yc[oidx], color='tomato',    lw=1.8, ls='--')
            ax_O.axhline(res['O_star'], color='k', ls='--', lw=1)
            ax_O.set_ylabel('O(t)')
            ax_O.legend(handles=[
                plt.Line2D([0], [0], color='steelblue', lw=1.8,
                           label='sensitized'),
                plt.Line2D([0], [0], color='tomato', lw=1.8, ls='--',
                           label='no ramp (control)'),
                plt.Line2D([0], [0], color='k', ls='--', lw=1,
                           label=f"O* = {res['O_star']:.3f}"),
                mpatches.Patch(facecolor=RAMP_COLOR, alpha=0.6,
                               label=(f"step → {res['primed_val']:.3g}"
                                      if res.get('fast_prep')
                                      else f"ramp → {res['primed_val']:.3g}")),
                mpatches.Patch(facecolor=TREAT_COLOR, alpha=0.6,
                               label=f"treat → {res['treatment_val']:.3g}"),
            ], fontsize=7.5, loc='upper right')
            ax_O.set_title(
                f"param = {param}  |  {v.replace('_', '→')}  "
                f"({res['base_val']:.3g} → {res['primed_val']:.3g} → "
                f"{res['treatment_val']:.3g})" + mark,
                fontsize=9,
                color='#1a7a1a' if is_correct else '#888888')
            plt.setp(ax_O.get_xticklabels(), visible=False)

            for ax_v, (vi, vname) in zip(ax_others, other_vars):
                ax_v.plot(t, y[vi],  color='steelblue', lw=1.8)
                ax_v.plot(t, yc[vi], color='tomato',    lw=1.8, ls='--')
                ax_v.set_ylabel(vname)
                plt.setp(ax_v.get_xticklabels(), visible=False)

            if has_cancer:
                c_sens = y[cancer_idx]
                c_ctrl = yc[cancer_idx]

                n_treat  = np.searchsorted(t, t_sw, side='right')
                min_sens = c_sens[n_treat:].min()
                min_ctrl = c_ctrl[n_treat:].min()
                ratio    = min_sens / min_ctrl if min_ctrl != 0 else float('nan')

                def ttp(cancer_traj):
                    cross = np.where(cancer_traj > cancer_threshold)[0]
                    return t[cross[0]] if len(cross) else float('inf')

                ttp_s = ttp(c_sens)
                ttp_c = ttp(c_ctrl)

                ax_C.plot(t, c_sens, color='steelblue', lw=1.8)
                ax_C.plot(t, c_ctrl, color='tomato',    lw=1.8, ls='--')
                ax_C.axhline(cancer_threshold, color='k', ls=':', lw=0.9, alpha=0.6)
                if np.isfinite(ttp_s):
                    ax_C.axvline(ttp_s, color='steelblue', ls=':', lw=1.2, alpha=0.8)
                if np.isfinite(ttp_c):
                    ax_C.axvline(ttp_c, color='tomato',    ls=':', lw=1.2, alpha=0.8)
                ax_C.set_ylabel('Cancer')

                ttp_s_str = '%.1f' % ttp_s if np.isfinite(ttp_s) else 'never'
                ttp_c_str = '%.1f' % ttp_c if np.isfinite(ttp_c) else 'never'
                stats = ('min ratio (s/c): %.3f\n'
                         'TTP sensitized: %s\n'
                         'TTP control:    %s') % (ratio, ttp_s_str, ttp_c_str)
                ax_C.text(0.97, 0.97, stats, transform=ax_C.transAxes,
                          fontsize=7.5, ha='right', va='top', family='monospace',
                          bbox=dict(facecolor='white', alpha=0.75, edgecolor='none',
                                    boxstyle='round,pad=0.3'))
                plt.setp(ax_C.get_xticklabels(), visible=False)

            p_ctrl = np.where(t <= t_sw, res['base_val'], res['treatment_val'])
            ax_p.plot(t, p,      color='steelblue', lw=1.5)
            ax_p.plot(t, p_ctrl, color='tomato',    lw=1.5, ls='--')
            ax_p.set_ylabel(param)
            ax_p.set_xlabel('time')
            yticks = sorted({res['base_val'], res['primed_val'], res['treatment_val']})
            ax_p.set_yticks(yticks)
            ax_p.set_yticklabels([f"{v:.3g}" for v in yticks], fontsize=7)

    fig.suptitle(f'Sensitization strategy — model {model.name}', fontsize=11)
    plt.show()

    INF_CAP = np.log2(20)
    if has_cancer:
        bar_params, lfc_cancer, lfc_ttp, lfc_O, lfc_O_is_inf, bar_variants = [], [], [], [], [], []

        for param in model.rpa_params:
            for v in variants_to_show:
                res  = ss.results[param][v]
                t_r  = res['t']
                y_r  = res['y']
                yc_r = res['y_ctrl']
                t_sw = res['ramp_duration']
                n_rp = np.searchsorted(t_r, t_sw, side='right')

                c_s = y_r[cancer_idx, n_rp:]
                c_c = yc_r[cancer_idx, n_rp:]
                o_s = y_r[model.output_idx, n_rp:]
                o_c = yc_r[model.output_idx, n_rp:]

                if v == 'up_down':
                    ext_cs, ext_cc = c_s.min(), c_c.min()
                    lfc_c = np.log2(ext_cs / ext_cc) if ext_cc > 0 and ext_cs > 0 else np.nan
                else:
                    ext_cs, ext_cc = c_s.max(), c_c.max()
                    lfc_c = np.log2(ext_cc / ext_cs) if ext_cc > 0 and ext_cs > 0 else np.nan

                def _ttp(traj, _t=t_r[n_rp:]):
                    cross = np.where(traj > cancer_threshold)[0]
                    return _t[cross[0]] if len(cross) else float('inf')

                ttp_s = _ttp(c_s)
                ttp_c = _ttp(c_c)
                if   np.isfinite(ttp_s) and np.isfinite(ttp_c) and ttp_c > 0:
                    lfc_t = np.log2(ttp_s / ttp_c)
                elif not np.isfinite(ttp_s) and np.isfinite(ttp_c):
                    lfc_t = INF_CAP
                elif np.isfinite(ttp_s) and not np.isfinite(ttp_c):
                    lfc_t = -INF_CAP
                else:
                    lfc_t = np.nan

                if v == 'up_down':
                    ext_os, ext_oc = o_s.min(), o_c.min()
                    if ext_oc <= 0 and ext_os > 0:
                        lfc_o, inf_o = INF_CAP, True
                    elif ext_os > 0 and ext_oc > 0:
                        lfc_o, inf_o = np.log2(ext_os / ext_oc), False
                    else:
                        lfc_o, inf_o = np.nan, False
                else:
                    ext_os, ext_oc = o_s.max(), o_c.max()
                    if ext_os <= 0 and ext_oc > 0:
                        lfc_o, inf_o = INF_CAP, True
                    elif ext_oc > 0 and ext_os > 0:
                        lfc_o, inf_o = np.log2(ext_oc / ext_os), False
                    else:
                        lfc_o, inf_o = np.nan, False

                bar_params.append(param)
                bar_variants.append(v)
                lfc_cancer.append(lfc_c)
                lfc_ttp.append(lfc_t)
                lfc_O.append(lfc_o)
                lfc_O_is_inf.append(inf_o)
                break

        if bar_params:
            x = np.arange(len(bar_params))
            w = max(6, 2.2 * len(bar_params))
            fig2, (ax_r, ax_t, ax_o) = plt.subplots(1, 3, figsize=(w * 1.5, 4))

            colors_r = ['#d7e5c5' if v > 0 else '#f6ceca' for v in lfc_cancer]
            ax_r.bar(x, lfc_cancer, color=colors_r, edgecolor='k', linewidth=0.6)
            ax_r.axhline(0, color='k', lw=1)
            ax_r.set_xticks(x); ax_r.set_xticklabels(bar_params, fontsize=9)
            _vnt0  = bar_variants[0] if bar_variants else 'up_down'
            _c_lbl = 'min Cancer (c/s)' if _vnt0 == 'up_down' else 'max Cancer (c/s)'
            _o_lbl = 'min output O (c/s)' if _vnt0 == 'up_down' else 'max output O (s/c)'
            ax_r.set_ylabel('log₂ fold-change')
            ax_r.set_title(_c_lbl + chr(10) + '(> 0 = helps)', fontsize=9)
            ax_r.spines['top'].set_visible(False); ax_r.spines['right'].set_visible(False)

            colors_t = ['#d7e5c5' if (not np.isnan(v) and v > 0) else '#f6ceca'
                        for v in lfc_ttp]
            ax_t.bar(x, lfc_ttp, color=colors_t, edgecolor='k', linewidth=0.6)
            ax_t.axhline(0, color='k', lw=1)
            for xi, v in zip(x, lfc_ttp):
                if np.isfinite(v) and abs(v) >= INF_CAP * 0.95:
                    ax_t.text(xi, v + 0.05 * np.sign(v), '∞',
                              ha='center', va='bottom' if v > 0 else 'top', fontsize=9)
            ax_t.set_xticks(x); ax_t.set_xticklabels(bar_params, fontsize=9)
            ax_t.set_ylabel('log₂ fold-change  (sens / ctrl)')
            ax_t.set_title('TTP' + chr(10) + '(> 0 = helps)', fontsize=9)
            ax_t.spines['top'].set_visible(False); ax_t.spines['right'].set_visible(False)

            lfc_O_disp = [np.clip(v, -INF_CAP, INF_CAP) for v in lfc_O]
            colors_o = ['#d7e5c5' if v > 0 else '#f6ceca' for v in lfc_O_disp]
            ax_o.bar(x, lfc_O_disp, color=colors_o, edgecolor='k', linewidth=0.6)
            ax_o.axhline(0, color='k', lw=1)
            for xi, v_disp, v_raw, is_inf in zip(x, lfc_O_disp, lfc_O, lfc_O_is_inf):
                if is_inf:
                    ax_o.text(xi, v_disp + 0.05 * np.sign(v_disp), '∞',
                              ha='center', va='bottom' if v_disp > 0 else 'top', fontsize=9)
                elif np.isfinite(v_raw) and abs(v_raw) > abs(v_disp) * 1.01:
                    ax_o.text(xi, v_disp + 0.05 * np.sign(v_disp),
                              f'▲{v_raw:.1f}',
                              ha='center', va='bottom' if v_disp > 0 else 'top', fontsize=8)
            ax_o.set_xticks(x); ax_o.set_xticklabels(bar_params, fontsize=9)
            ax_o.set_ylabel('log₂ fold-change')
            ax_o.set_title(_o_lbl + chr(10) + '(> 0 = helps)', fontsize=9)
            ax_o.spines['top'].set_visible(False); ax_o.spines['right'].set_visible(False)

            fig2.suptitle('Sensitization metrics — model ' + model.name +
                          '  (correct variant only)', fontsize=10)
            plt.tight_layout()
            plt.show()

    return ss
