import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class SeparationOfTimescalesStrategy:
    """
    Periodic intermittent treatment for an RPAModel based on separation of timescales.

    For each RPA parameter p two variants are tried (2× and 0.5× baseline).
    Treatment timing (T_on, T_off) is determined automatically:

        T_on  — simulate constant treatment for the full t_max_ton window; T_on is
                the time at which Cancer reaches its minimum.  If Cancer never drops
                below its initial value, the (param, scale) pair is marked
                non-successful and skipped.
        T_off — apply one T_on pulse then switch off; T_off is the first time ALL
                homeostatic variables (every state var except O and Cancer) satisfy
                |x_i(t) − x_i*| / |x_i*| <= 1 − recovery_frac.

    A periodic simulation of n_cycles × (T_on on, T_off off) is then produced.

    results[param][scale] keys
    --------------------------
    success                  : bool — False when Cancer never decays under treatment
    T_on, T_off              : timing floats  (only when success=True)
    t_ton_diag, y_ton_diag  : constant-treatment diagnostic trajectory
    ton_min_i                : array index of Cancer minimum (success=True only)
    t_toff_diag, y_toff_diag : T_on pulse + recovery trajectory (success=True only)
    toff_rec_times           : per-homeostatic-var recovery times relative to
                               recovery start (success=True only)
    t_periodic, y_periodic, on_mask : periodic simulation (success=True only)
    O_star, param_val, base_val, recovery_frac
    """

    def __init__(self, model):
        self.model    = model
        self.baseline = dict(model.params_base)
        self._oidx    = model.output_idx
        _extra_names       = set(getattr(model, '_extra_vars', []))
        self._hidx         = [i for i in range(len(model.state_vars))
                              if i != self._oidx
                              and model.state_vars[i] not in _extra_names]
        self._cancer_idx   = next(
            (i for i, v in enumerate(model.state_vars) if v in _extra_names), None)
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

    def _init_y0(self, overrides=None):
        """SS with extra vars (e.g. Cancer) initialized to 1.0."""
        y0 = self._eval_ss(overrides)
        m  = self.model
        if getattr(m, '_extra_vars', None):
            y0[len(m.state_vars) - len(m._extra_vars):] = 1.0
        return y0

    @staticmethod
    def _rk4_step(f, t, y, dt):
        k1 = np.array(f(t,          y))
        k2 = np.array(f(t + 0.5*dt, y + 0.5*dt*k1))
        k3 = np.array(f(t + 0.5*dt, y + 0.5*dt*k2))
        k4 = np.array(f(t + dt,     y +      dt*k3))
        return y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def _integrate(self, f, y0, t_max, dt):
        n     = int(round(t_max / dt))
        t_arr = np.linspace(0, t_max, n + 1)
        y_arr = np.zeros((len(y0), n + 1))
        y_arr[:, 0] = y0
        for i in range(n):
            y_arr[:, i + 1] = self._rk4_step(f, t_arr[i], y_arr[:, i], dt)
        return t_arr, y_arr

    def find_T_on(self, param, scale_val, dt=0.005, t_max=200):
        """
        Simulate constant treatment for t_max; T_on = time of Cancer minimum.
        Returns (t_arr, y_arr, T_on, min_i, success).
        success=False when Cancer never drops below its initial value.
        """
        y0  = self._init_y0()
        f   = self.model.get_f({param: scale_val})
        t_arr, y_arr = self._integrate(f, y0, t_max, dt)

        cidx        = self._cancer_idx
        cancer      = y_arr[cidx]
        cancer_init = cancer[0]
        min_i       = int(np.argmin(cancer))

        if cancer[min_i] >= cancer_init:
            return t_arr, y_arr, None, None, False

        return t_arr, y_arr, float(t_arr[min_i]), min_i, True

    def find_T_off(self, param, scale_val, T_on, recovery_frac=0.9,
                   dt=0.005, t_max=500):
        """
        Apply T_on pulse then switch off; T_off is the first time ALL homeostatic
        variables satisfy |x_i(t) − x_i*| / |x_i*| <= 1 − recovery_frac.
        Returns (t_combined, y_combined, rec_times, T_off).
        """
        y0_ss   = self._init_y0()
        f_treat = self.model.get_f({param: scale_val})
        f_base  = self.model.get_f()

        t_on_arr, y_on = self._integrate(f_treat, y0_ss, T_on, dt)
        t_rec,    y_rec = self._integrate(f_base, y_on[:, -1], t_max, dt)

        hidx      = self._hidx
        ss_h      = y0_ss[hidx]
        threshold = 1.0 - recovery_frac
        rec_times = np.full(len(hidx), np.nan)

        for k, hi in enumerate(hidx):
            rel_err = np.abs(y_rec[hi] - ss_h[k]) / (np.abs(ss_h[k]) + 1e-12)
            cross   = np.where(rel_err <= threshold)[0]
            if len(cross):
                rec_times[k] = float(t_rec[cross[0]])

        T_off  = float(np.nanmax(rec_times)) if not np.all(np.isnan(rec_times)) else t_max
        t_comb = np.concatenate([t_on_arr, t_rec[1:] + T_on])
        y_comb = np.concatenate([y_on, y_rec[:, 1:]], axis=1)

        return t_comb, y_comb, rec_times, T_off

    def simulate(self, param, scale='2x', n_cycles=5, recovery_frac=0.9,
                 dt=0.005, t_max_ton=200, t_max_toff=500):
        """Determine T_on / T_off and simulate periodic treatment."""
        base_val  = self.baseline[param]
        scale_val = base_val * (2.0 if scale == '2x' else 0.5)

        t_ton, y_ton, T_on, min_i, success = self.find_T_on(
            param, scale_val, dt=dt, t_max=t_max_ton)

        if not success:
            self.results.setdefault(param, {})[scale] = dict(
                success=False,
                t_ton_diag=t_ton, y_ton_diag=y_ton,
                O_star=self.O_star, param_val=scale_val, base_val=base_val,
            )
            return None, None

        t_toff, y_toff, rec_times, T_off = self.find_T_off(
            param, scale_val, T_on, recovery_frac=recovery_frac,
            dt=dt, t_max=t_max_toff)

        f_treat = self.model.get_f({param: scale_val})
        f_base  = self.model.get_f()
        n_on    = int(round(T_on  / dt))
        n_off   = int(round(T_off / dt))
        n_total = n_cycles * (n_on + n_off)

        y0      = self._init_y0()
        t_per   = np.zeros(n_total + 1)
        y_per   = np.zeros((len(y0), n_total + 1))
        on_mask = np.zeros(n_total + 1, dtype=bool)
        y_per[:, 0] = y0

        step  = 0
        t_cur = 0.0
        for _ in range(n_cycles):
            for f_cur, is_on, n_phase in [
                (f_treat, True,  n_on),
                (f_base,  False, n_off),
            ]:
                for _ in range(n_phase):
                    on_mask[step]      = is_on
                    y_per[:, step + 1] = self._rk4_step(f_cur, t_cur, y_per[:, step], dt)
                    t_cur             += dt
                    t_per[step + 1]    = t_cur
                    step              += 1
        on_mask[step] = on_mask[step - 1]

        self.results.setdefault(param, {})[scale] = dict(
            success=True,
            T_on=T_on, T_off=T_off,
            t_ton_diag=t_ton, y_ton_diag=y_ton, ton_min_i=min_i,
            t_toff_diag=t_toff, y_toff_diag=y_toff, toff_rec_times=rec_times,
            recovery_frac=recovery_frac,
            t_periodic=t_per[:step + 1], y_periodic=y_per[:, :step + 1],
            on_mask=on_mask[:step + 1],
            O_star=self.O_star, param_val=scale_val, base_val=base_val,
        )
        return T_on, T_off

    def simulate_all(self, **kwargs):
        for param in self.model.rpa_params:
            for scale in ('2x', '0.5x'):
                print(f"  param={param}  scale={scale} ...")
                T_on, T_off = self.simulate(param, scale=scale, **kwargs)
                if T_on is None:
                    print(f"    → non-successful (Cancer does not decay)")
                else:
                    print(f"    T_on = {T_on:.3f},  T_off = {T_off:.3f}")
        print("Done.")


def run_sot_demo(model, n_cycles=5, recovery_frac=0.9, dt=0.005,
                 t_max_ton=200, t_max_toff=500):
    """
    Build a SeparationOfTimescalesStrategy, simulate both scales for every
    RPA parameter, and plot results.

    For each (param, scale) panel (n_params rows × 2 columns):
      Row 0 — T_on diagnostic: Cancer(t) under constant treatment; dot at Cancer min.
      Row 1 — T_off diagnostic: homeostatic vars during T_on pulse + recovery.
      Row 2 — Periodic O(t): n_cycles of (T_on on, T_off off) with shaded on-periods.
      Row 3 — Periodic Cancer(t): tumor trajectory over the same cycles.
    Non-successful panels are labelled accordingly.
    """
    sot = SeparationOfTimescalesStrategy(model)
    print(f"Model {model.name}  |  O* = {sot.O_star:.4f}  |  "
          f"RPA params: {model.rpa_params}\n")
    sot.simulate_all(n_cycles=n_cycles, recovery_frac=recovery_frac, dt=dt,
                     t_max_ton=t_max_ton, t_max_toff=t_max_toff)

    ON_COLOR   = '#f0d8c8'
    has_cancer = sot._cancer_idx is not None
    n_inner    = 4 if has_cancer else 3
    h_ratios   = [1, 1.5, 1.2, 0.8] if has_cancer else [1, 1.5, 1.5]
    n          = len(model.rpa_params)
    fig_h      = (10 if has_cancer else 8) * n
    fig        = plt.figure(figsize=(14, fig_h))
    outer      = fig.add_gridspec(n, 2, hspace=0.65, wspace=0.35)

    for row, param in enumerate(model.rpa_params):
        for col, scale in enumerate(('2x', '0.5x')):
            res  = sot.results[param][scale]
            oidx = model.output_idx
            cidx = sot._cancer_idx
            hidx = sot._hidx

            inner     = outer[row, col].subgridspec(n_inner, 1, hspace=0.6,
                                                    height_ratios=h_ratios)
            ax_ton    = fig.add_subplot(inner[0])
            ax_toff   = fig.add_subplot(inner[1])
            ax_per    = fig.add_subplot(inner[2])
            ax_cancer = fig.add_subplot(inner[3], sharex=ax_per) if has_cancer else None

            if not res['success']:
                t_d, y_d = res['t_ton_diag'], res['y_ton_diag']
                ax_ton.plot(t_d, y_d[cidx], color='C3', lw=1.5)
                ax_ton.set_title(
                    f"param = {param} = {res['param_val']:.3g} ({scale})  "
                    f"— Cancer does not decay",
                    fontsize=8, color='gray')
                ax_ton.set_ylabel('Cancer(t)', fontsize=8)
                ax_ton.spines['top'].set_visible(False)
                ax_ton.spines['right'].set_visible(False)
                for ax in [ax_toff, ax_per] + ([ax_cancer] if has_cancer else []):
                    ax.text(0.5, 0.5, 'non-successful',
                            ha='center', va='center', transform=ax.transAxes,
                            fontsize=9, color='gray', style='italic')
                    ax.set_xticks([]); ax.set_yticks([])
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                continue

            T_on  = res['T_on']
            T_off = res['T_off']

            t_d, y_d = res['t_ton_diag'], res['y_ton_diag']
            min_i    = res['ton_min_i']

            ax_ton.plot(t_d, y_d[cidx], color='C3', lw=1.5)
            ax_ton.axvline(T_on, color='C1', ls='--', lw=1.2)
            ax_ton.scatter([T_on], [y_d[cidx, min_i]], color='C1', zorder=5, s=45)
            ax_ton.set_title(
                f"T_on diagnostic — {param} = {res['param_val']:.3g} ({scale})  "
                f"|  T_on = {T_on:.3f}",
                fontsize=8)
            ax_ton.set_ylabel('Cancer(t)', fontsize=8)
            ax_ton.legend(handles=[
                plt.Line2D([0], [0], color='C1', ls='--', lw=1.2,
                           label=f'T_on = {T_on:.3f}  (Cancer min)'),
            ], fontsize=7, frameon=False)
            ax_ton.spines['top'].set_visible(False)
            ax_ton.spines['right'].set_visible(False)

            t_r, y_r  = res['t_toff_diag'], res['y_toff_diag']
            rec_times = res['toff_rec_times']
            n_hv      = len(hidx)
            cols_h    = plt.cm.tab10(np.linspace(0, 0.9, n_hv))
            y0_ss     = sot._init_y0()

            ax_toff.axvspan(0, T_on, color=ON_COLOR, alpha=0.45, lw=0)
            ax_toff.axvline(T_on, color='k', ls=':', lw=0.9)
            ax_toff.axvline(T_on + T_off, color='k', ls='--', lw=1.2)

            for k, hi in enumerate(hidx):
                vname = model.state_vars[hi]
                ax_toff.plot(t_r, y_r[hi], color=cols_h[k], lw=1.2, label=vname)
                ax_toff.axhline(y0_ss[hi], color=cols_h[k], ls=':', lw=0.7, alpha=0.5)
                if not np.isnan(rec_times[k]):
                    t_abs = T_on + rec_times[k]
                    i_abs = np.searchsorted(t_r, t_abs, side='right') - 1
                    ax_toff.axvline(t_abs, color=cols_h[k], ls='--', lw=0.8, alpha=0.75)
                    ax_toff.scatter([t_abs], [y_r[hi, i_abs]],
                                    color=cols_h[k], zorder=5, s=28)

            ax_toff.set_title(
                f"T_off diagnostic  (recovery >= {res['recovery_frac']:.0%})  "
                f"|  T_off = {T_off:.3f}",
                fontsize=8)
            ax_toff.set_ylabel('homeostatic vars', fontsize=8)
            ax_toff.set_xlabel('time', fontsize=8)
            ax_toff.legend(
                handles=[
                    *[plt.Line2D([0], [0], color=cols_h[k], lw=1.2,
                                 label=model.state_vars[hidx[k]])
                      for k in range(n_hv)],
                    mpatches.Patch(facecolor=ON_COLOR, alpha=0.55, label='treatment on'),
                    plt.Line2D([0], [0], color='k', ls='--', lw=1.2,
                               label=f'T_off = {T_off:.3f}'),
                ],
                fontsize=6.5, frameon=False, ncol=2)
            ax_toff.spines['top'].set_visible(False)
            ax_toff.spines['right'].set_visible(False)

            t_p, y_p = res['t_periodic'], res['y_periodic']
            mask     = res['on_mask']

            changes = np.where(np.diff(mask.astype(int)) != 0)[0]
            bounds  = np.concatenate([[0], changes + 1, [len(t_p) - 1]])
            for j in range(len(bounds) - 1):
                i0, i1 = bounds[j], bounds[j + 1]
                if mask[i0]:
                    ax_per.axvspan(t_p[i0], t_p[i1], color=ON_COLOR, alpha=0.4, lw=0)

            ax_per.plot(t_p, y_p[oidx], color='steelblue', lw=1.5)
            ax_per.axhline(res['O_star'], color='k', ls='--', lw=1)
            ax_per.set_title(
                f"Periodic treatment  ({n_cycles} cycles, "
                f"T_on = {T_on:.3f}, T_off = {T_off:.3f})",
                fontsize=8)
            ax_per.set_ylabel('O(t)', fontsize=8)
            ax_per.legend(handles=[
                plt.Line2D([0], [0], color='steelblue', lw=1.5, label='O(t)'),
                plt.Line2D([0], [0], color='k', ls='--', lw=1,
                           label=f"O* = {res['O_star']:.3f}"),
                mpatches.Patch(facecolor=ON_COLOR, alpha=0.5, label='treatment on'),
            ], fontsize=7, frameon=False)
            ax_per.spines['top'].set_visible(False)
            ax_per.spines['right'].set_visible(False)
            if has_cancer:
                plt.setp(ax_per.get_xticklabels(), visible=False)
            else:
                ax_per.set_xlabel('time', fontsize=8)

            if has_cancer:
                for j in range(len(bounds) - 1):
                    i0, i1 = bounds[j], bounds[j + 1]
                    if mask[i0]:
                        ax_cancer.axvspan(t_p[i0], t_p[i1],
                                          color=ON_COLOR, alpha=0.4, lw=0)
                ax_cancer.plot(t_p, y_p[cidx], color='C3', lw=1.5)
                ax_cancer.set_ylabel('Cancer', fontsize=8)
                ax_cancer.set_xlabel('time', fontsize=8)
                ax_cancer.legend(handles=[
                    plt.Line2D([0], [0], color='C3', lw=1.5, label='Cancer(t)'),
                    mpatches.Patch(facecolor=ON_COLOR, alpha=0.5, label='treatment on'),
                ], fontsize=7, frameon=False)
                ax_cancer.spines['top'].set_visible(False)
                ax_cancer.spines['right'].set_visible(False)

    fig.suptitle(f'Separation-of-timescales strategy — model {model.name}', fontsize=11)
    plt.tight_layout()
    plt.show()
    return sot
