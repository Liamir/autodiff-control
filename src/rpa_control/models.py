import numpy as np
from scipy.integrate import solve_ivp
from sympy import symbols, solve, simplify, diff, lambdify, Integer


class RPAModel:
    """
    RPA ODE model defined symbolically with SymPy.

    Parameters
    ----------
    name        : short label
    state_vars  : ordered list of state-variable name strings
    output_var  : name of the output variable (must appear in state_vars)
    params_base : dict of base values for every parameter, including 'Input'
    equations   : list of sympy expressions — the RHS of dxi/dt in the same
                  order as state_vars, written using symbols whose names match
                  state_vars and params_base keys (all declared positive=True)
    extra_vars  : optional list of extra variable name strings appended after
                  the homeostatic state vars (e.g. ['Cancer']).  Their SS is
                  set to 0.  They are excluded from the sympy solve.
    extra_f     : callable(t, y_full, O_star_num) -> list of derivatives for
                  the extra variables, evaluated at every ODE step.

    Attributes set automatically
    ----------------------------
    O_star_sym  : sympy expression for the analytic steady-state of output_var
    O_star_num  : float — numerical O* at baseline parameters
    rpa_params  : list of parameter names with ∂O*/∂k = 0
    """

    def __init__(self, name, state_vars, output_var, params_base, equations,
                 extra_vars=None, extra_f=None):
        self.name        = name
        self.state_vars  = list(state_vars)
        self.output_var  = output_var
        self.output_idx  = self.state_vars.index(output_var)
        self.params_base = dict(params_base)
        self._equations  = list(equations)

        self._sym_state   = [symbols(v, positive=True) for v in state_vars]
        self._param_names = list(params_base.keys())
        self._sym_params  = [symbols(k, positive=True) for k in self._param_names]

        self._ss_dict   = self._solve_ss()
        O_sym           = self._sym_state[self.output_idx]
        self.O_star_sym = simplify(self._ss_dict[O_sym])

        self.rpa_params = [
            name for name, sym in zip(self._param_names, self._sym_params)
            if diff(self.O_star_sym, sym) == 0
        ]

        _args       = self._sym_state + self._sym_params
        self._f_lam = lambdify(_args, self._equations, modules='numpy')

        self._extra_vars = list(extra_vars) if extra_vars else []
        self._extra_f    = extra_f

        _base_subs = {sym: self.params_base[name]
                      for name, sym in zip(self._param_names, self._sym_params)}
        self.O_star_num = float(self.O_star_sym.subs(_base_subs))

        if self._extra_vars:
            self.state_vars = self.state_vars + self._extra_vars
            for vname in self._extra_vars:
                vsym = symbols(vname, positive=True)
                self._sym_state.append(vsym)
                self._ss_dict[vsym] = Integer(0)

    def _solve_ss(self):
        sols = solve(self._equations, self._sym_state, dict=True)
        if not sols:
            raise ValueError(f"[{self.name}] No analytic steady-state found.")
        return sols[0]

    def get_f(self, rpa_overrides=None):
        """Return f(t, y) -> dydt for scipy.integrate.solve_ivp."""
        p     = dict(self.params_base)
        if rpa_overrides:
            p.update(rpa_overrides)
        pvals   = [p[k] for k in self._param_names]
        f_lam   = self._f_lam
        extra_f = self._extra_f
        O_star  = self.O_star_num
        n_base  = len(self.state_vars) - len(self._extra_vars)

        if extra_f is None:
            def f(t, y):
                return f_lam(*y, *pvals)
        else:
            def f(t, y):
                return list(f_lam(*y[:n_base], *pvals)) + list(extra_f(t, y, O_star))
        return f

    def _repr_latex_(self):
        from sympy import latex as sym_latex

        ode_rows = r' \\'.join(
            r'\dot{' + sym_latex(sv) + r'} &= ' + sym_latex(eq)
            for sv, eq in zip(self._sym_state, self._equations)
        )

        O_sym   = self._sym_state[self.output_idx]
        ss_row  = sym_latex(O_sym) + r'^{*} &= ' + sym_latex(self.O_star_sym)

        rpa_str = r',\;'.join(
            sym_latex(symbols(p, positive=True))
            for p in self.rpa_params
        )
        rpa_row = r'\text{RPA params} &:\; ' + rpa_str

        body = (
            r'& \textbf{Case\;' + self.name + r'} \\[4pt]'
            + ode_rows
            + r' \\[6pt]'
            + ss_row
            + r' \\[6pt]'
            + rpa_row
        )
        return r'$$\begin{aligned}' + body + r'\end{aligned}$$'

    def __repr__(self):
        return (f"RPAModel('{self.name}', output='{self.output_var}', "
                f"rpa_params={self.rpa_params})")


# ── Shared symbols ────────────────────────────────────────────────────────────
Iv, O, O1, O2, O3 = symbols('I  O  O1  O2  O3', positive=True)
B,  C,  X,  X1, X2 = symbols('B  C  X   X1  X2', positive=True)

Input_sym = symbols('Input', positive=True)

(k1, k2, k3, k4, k5, k6, k7, k8, k9,
 k10, k11, k12, k13, k14, k15, k16, k17, k18) = symbols(
    'k1 k2 k3 k4 k5 k6 k7 k8 k9 k10 k11 k12 k13 k14 k15 k16 k17 k18',
    positive=True)

# ── Cancer readout ────────────────────────────────────────────────────────────
c1_cancer = 0.05
c2_cancer = 1.0

def make_cancer_f(oidx, c1=c1_cancer, c2=c2_cancer, decay_below=True):
    """Return extra_f for Cancer given the output variable index oidx."""
    def cancer_f(t, y, O_star):
        if decay_below:
            return [y[-1] * (c1 - c2 * max(O_star - y[oidx], 0.0))]
        else: # decay when output is above O*, e.g. for IFFL
            return [y[-1] * (c1 - c2 * max(y[oidx] - O_star, 0.0))]
    return cancer_f


# ── Case a ────────────────────────────────────────────────────────────────────
model_a = RPAModel(
    name='a',
    state_vars=['I', 'O', 'O1', 'C', 'B'],
    output_var='O',
    params_base=dict(Input=1.0,
                     k1=1.0, k2=0.5, k3=1.0, k4=0.5, k5=0.5, k6=1.0,
                     k7=1.0, k8=0.5, k9=0.5, k10=0.5, k11=0.5, k12=0.5),
    equations=[
        k1*Input_sym - k2*Iv  - k3*O1,
        k4*Iv        + k5*C   - k6*O,
        k7*Iv        - k8,
        k9*O1        - k10*B*C,
        k11*O1       - k12*B,
    ],
    extra_vars=['Cancer'], extra_f=make_cancer_f(1),
)

# ── Case b ────────────────────────────────────────────────────────────────────
model_b = RPAModel(
    name='b',
    state_vars=['I', 'B', 'X', 'O1', 'O'],
    output_var='O',
    params_base=dict(Input=1.0,
                     k1=1.0, k2=1.0, k3=1.0, k4=1.0, k5=0.5,
                     k7=1.0, k8=1.0, k9=0.5, k10=1.0, k11=1.0),
    equations=[
        k1*Input_sym - k2*Iv,
        k3*Iv        - k4*B,
        k5*B / O1    - k7*X,
        k8*X         - k9,
        k10*Iv*X     - k11*O*B,
    ],
    extra_vars=['Cancer'], extra_f=make_cancer_f(4),
)

# ── Case c ────────────────────────────────────────────────────────────────────
model_c = RPAModel(
    name='c',
    state_vars=['I', 'O', 'O1', 'X', 'O2'],
    output_var='O',
    params_base=dict(Input=1.0,
                     k1=1.0, k2=0.5, k3=1.0, k4=0.5, k5=0.5, k6=1.0,
                     k7=1.0, k8=0.5, k9=0.5, k10=0.5, k11=0.5, k12=1.0, k13=0.5),
    equations=[
        k1*Input_sym - k2*Iv  - k3*O1,
        k4*Iv        + k5*X   - k6*O,
        k7*Iv        - k8,
        k9*O1        - k10*X  - k11*O2,
        k12*X        - k13,
    ],
    extra_vars=['Cancer'], extra_f=make_cancer_f(1),
)

# ── Case d ────────────────────────────────────────────────────────────────────
model_d = RPAModel(
    name='d',
    state_vars=['I', 'O', 'O1', 'X', 'O2'],
    output_var='O',
    params_base=dict(Input=1.0,
                     k1=0.5, k2=0.5, k3=0.5, k4=1.0, k5=1.0, k6=1.0,
                     k7=1.0, k8=1.0, k9=0.5, k10=1.0, k11=1.0, k12=0.5),
    equations=[
        k1*Input_sym - k2*Iv  - k3*O1,
        k4*Iv        - k5*O,
        k11*X        - k12,
        k8*O2        - k9*X   - k10*O1,
        k6*O         - k7*X,
    ],
    extra_vars=['Cancer'], extra_f=make_cancer_f(1),
)

# ── Case e ────────────────────────────────────────────────────────────────────
model_e = RPAModel(
    name='e',
    state_vars=['I', 'O1', 'X', 'O2', 'B', 'C', 'O'],
    output_var='O',
    params_base=dict(Input=1.0,
                     k1=0.5, k2=1.0, k4=1.0, k5=1.0, k6=1.0, k7=1.0, k8=0.4,
                     k9=1.0, k10=0.5, k11=1.0, k12=1.0, k14=0.5, k15=1.0,
                     k16=0.5, k17=0.5, k18=1.0),
    equations=[
        k1*Input_sym - k2*O1*Iv,
        k4*Iv        - k5*X,
        k6*O1        - k7*O2  - k8*X,
        k9*X         - k10,
        k11*O2       - k12*B,
        k14*O2       - k15*B*C,
        k16*Iv       + k17*C  - k18*O,
    ],
    extra_vars=['Cancer'], extra_f=make_cancer_f(6),
)

# ── Case f ────────────────────────────────────────────────────────────────────
model_f = RPAModel(
    name='f',
    state_vars=['I', 'O1', 'X1', 'O2', 'X2', 'O3', 'O'],
    output_var='O',
    params_base=dict(Input=1.0,
                     k1=0.5, k2=0.5, k3=0.5, k4=1.0, k5=1.0, k6=1.0, k7=1.0,
                     k8=1.0, k9=1.0, k10=0.5, k11=1.0, k12=0.5, k13=1.0,
                     k14=1.0, k15=0.5, k16=0.5, k17=0.5, k18=1.0),
    equations=[
        k1*Input_sym - k2*Iv   - k3*O1,
        k4*Iv        - k5*X1,
        k6*O1        - k7*O2   - k8*X1,
        k9*X1        - k10,
        k11*O2       - k12*X2  - k13*O3,
        k14*X2       - k15,
        k16*Iv       + k17*X2  - k18*O,
    ],
    extra_vars=['Cancer'], extra_f=make_cancer_f(6),
)

ALL_MODELS = [model_a, model_b, model_c, model_d, model_e, model_f]

# ── Integral-feedback model ───────────────────────────────────────────────────
A_sym               = symbols('A',               positive=True)
alpha, beta1, beta2 = symbols('alpha beta1 beta2', positive=True)
mu_sym              = symbols('mu',              positive=True)

# model_if = RPAModel(
#     name='IF',
#     state_vars=['A', 'B'],
#     output_var='B',
#     params_base=dict(mu=1.0, alpha=10.0, beta1=10.0, beta2=1.0),
#     equations=[
#         alpha * (mu_sym - B),
#         beta1 * A_sym - beta2 * B,
#     ],
#     extra_vars=['Cancer'], extra_f=make_cancer_f(1, c2=100.0),
# )

model_if = RPAModel(
    name='IF',
    state_vars=['A', 'B'],
    output_var='B',
    params_base=dict(mu=1.0, alpha=1.0, beta1=1.0, beta2=1.0),
    equations=[
        alpha * (mu_sym - B),
        beta1 * A_sym - beta2 * B,
    ],
    extra_vars=['Cancer'], extra_f=make_cancer_f(1, decay_below=True),
)

# ── IFFL model ────────────────────────────────────────────────────────────────
x_s, y_s                 = symbols('x y',             positive=True)
delta_s, beta_s, gamma_s = symbols('delta beta gamma', positive=True)

model_iffl = RPAModel(
    name='IFFL',
    state_vars=['x', 'y'],
    output_var='y',
    params_base=dict(Input=1.0, alpha=1.0, delta=1.0, beta=1.0, gamma=1.0),
    equations=[
        alpha   * Input_sym - delta_s * x_s,
        beta_s  * Input_sym - gamma_s * x_s * y_s,
    ],
    extra_vars=['Cancer'], extra_f=make_cancer_f(1, c1=0.025),
)

# ── ABC model ─────────────────────────────────────────────────────────────────
mu    = 1.0
a1    = 1.0
b1    = 1.0
b1_on = b1 / 2
b2    = 2.0
_c1   = 0.05
_c2   = 1.0

B_ss    = mu
A_ss    = b2 * mu / b1
A_ss_on = b2 * mu / b1_on


def abc_rhs(t, y, b1_val):
    A, _B, _C = y
    dA = a1 * (mu - _B)
    dB = b1_val * A - b2 * _B
    dC = _C * (_c1 - _c2 * max(mu - _B, 0.0))
    return [dA, dB, dC]


class ABCModel:
    """
    Thin wrapper for abc_rhs compatible with simulate_rpa_on_off / simulate_rpa_intermittent.

    The only supported parameter_name is 'b1'. Pass y0 explicitly when calling
    simulation functions (C has no finite steady state).
    """
    name       = 'ABC'
    state_vars = ['A', 'B', 'C']

    def get_f(self, overrides=None):
        b1_val = overrides.get('b1', b1) if overrides else b1
        return lambda t, y: abc_rhs(t, y, b1_val)


model_abc = ABCModel()


# ── Bare AB homeostatic pair (no tumor) ───────────────────────────────────────
class ABModel:
    """
    Just the A, B homeostatic pair -- no tumor variable. Unlike ABCModel,
    a1/b2/mu/b1 are instance attributes (defaulting to the module-level
    values above), so different AB parameterizations can be explored side
    by side. Compatible with simulate_rpa_on_off / simulate_rpa_intermittent.

    The only supported parameter_name is 'b1'.
    """
    name       = 'AB'
    state_vars = ['A', 'B']

    def __init__(self, a1=a1, b2=b2, mu=mu, b1=b1):
        self.a1 = a1
        self.b2 = b2
        self.mu = mu
        self.b1 = b1

    def steady_state(self, b1_val=None):
        """Off-treatment (or given b1_val) fixed point [A*, B*]."""
        b1_val = self.b1 if b1_val is None else b1_val
        return np.array([self.b2 * self.mu / b1_val, self.mu])

    def get_f(self, overrides=None):
        b1_val = overrides.get('b1', self.b1) if overrides else self.b1
        a1_, b2_, mu_ = self.a1, self.b2, self.mu

        def f(t, y):
            A, B = y
            return [a1_ * (mu_ - B), b1_val * A - b2_ * B]
        return f


# ── ABC model + two-clone competition (clonal selection) ─────────────────────
def abc_clonal_rhs(t, y, b1_val, r_s, r_r, carrying_capacity, cr_sensitivity=0.0, c2=_c2):
    """ABC homeostatic pair (A, B) driving therapy-induced decay of a
    drug-sensitive clone Cs, which competes logistically for one carrying
    capacity against a resistant clone Cr.

    cr_sensitivity in [0, 1] sets how much of Cs's kill rate still applies to
    Cr: 0 = fully resistant/immune (Cr never killed), values in (0,1) =
    partial resistance (Cr is still killed, just less efficiently than Cs).
    c2 sets the overall kill strength (shrinkage rate); defaults to the
    module-level base value."""
    A, _B, Cs, Cr = y
    dA = a1 * (mu - _B)
    dB = b1_val * A - b2 * _B

    kill = c2 * max(mu - _B, 0.0)
    growth_factor = 1 - (Cs + Cr) / carrying_capacity
    dCs = r_s * Cs * growth_factor - kill * Cs
    dCr = r_r * Cr * growth_factor - cr_sensitivity * kill * Cr
    return [dA, dB, dCs, dCr]


class ABCClonalModel:
    """
    ABC timescale-separation model (see ABCModel) combined with two-clone
    logistic competition (see SI_strategies_with_clonal_selection.ipynb):
    the sensitive clone Cs is killed at full strength while B is suppressed,
    the resistant clone Cr at a reduced strength set by cr_sensitivity
    (0 = fully resistant, as originally implemented), and Cs, Cr compete for
    one shared carrying capacity. Sparing some Cs by pulsing treatment
    (instead of dosing continuously) lets it keep competitively suppressing
    Cr, delaying resistant takeover.

    The only supported parameter_name is 'b1'. Pass y0 explicitly when calling
    simulate_rpa_on_off / simulate_rpa_intermittent (Cs, Cr have no finite
    steady state).
    """
    name       = 'ABC+clonal'
    state_vars = ['A', 'B', 'Cs', 'Cr']

    def __init__(self, r_s=0.5, r_r=0.2, carrying_capacity=1000.0, cr_sensitivity=0.0, c2=_c2):
        self.r_s = r_s
        self.r_r = r_r
        self.carrying_capacity = carrying_capacity
        self.cr_sensitivity = cr_sensitivity
        self.c2 = c2

    def get_f(self, overrides=None):
        b1_val = overrides.get('b1', b1) if overrides else b1
        return lambda t, y: abc_clonal_rhs(t, y, b1_val, self.r_s, self.r_r,
                                           self.carrying_capacity, self.cr_sensitivity,
                                           self.c2)


model_abc_clonal = ABCClonalModel()


# ── Feedback wrapper (purely numerical) ──────────────────────────────────────
class FeedbackModel:
    """
    Wraps an existing model (RPAModel, ABCModel, ...), adding a linear
    feedback term  sign * coeff * y[feedback_var]  into target_var's
    derivative. Purely numerical: base_model.get_f() is called as-is and the
    feedback term is added to the returned derivative vector, so no sympy
    solve/lambdify is involved and the base model is left untouched.

    No analytic steady state is available (there is no self._ss_dict).
    rpa_params defaults to the base model's list, since the feedback term
    only affects the returned derivative -- not the analytic homeostatic
    fixed point, which is unaffected as long as target_var's own equation
    doesn't define the output's steady state (verify per case, e.g. with
    simulation.classify_rpa_params, rather than assuming it always holds).

    coeff_name is added to params_base (default coeff_val), so the feedback
    strength can be swept without rebuilding the model -- either via
    get_f(overrides={coeff_name: ...}) or by mutating params_base directly.
    """

    def __init__(self, base_model, target_var, feedback_var,
                 coeff_name='b_fb', sign=1.0, coeff_val=0.5, rpa_params=None):
        self.base_model  = base_model
        self.name        = f'{base_model.name}+fb({feedback_var}->{target_var})'
        self.state_vars  = list(base_model.state_vars)
        self.output_var  = base_model.output_var
        self.output_idx  = base_model.output_idx
        self._extra_vars = list(getattr(base_model, '_extra_vars', []))
        self.params_base = {**base_model.params_base, coeff_name: coeff_val}
        self.rpa_params  = (list(rpa_params) if rpa_params is not None
                            else list(base_model.rpa_params))

        self._target_idx   = self.state_vars.index(target_var)
        self._feedback_idx = self.state_vars.index(feedback_var)
        self._coeff_name   = coeff_name
        self._sign         = sign

    def get_f(self, overrides=None):
        p = dict(self.params_base)
        if overrides:
            p.update(overrides)
        coeff       = p.pop(self._coeff_name)
        base_kwargs = {k: v for k, v in p.items() if k in self.base_model.params_base}
        f_base      = self.base_model.get_f(base_kwargs)
        ti, fi, sign = self._target_idx, self._feedback_idx, self._sign

        def f(t, y):
            dy      = list(f_base(t, y))
            dy[ti] += sign * coeff * y[fi]
            return dy
        return f


# ── ABC feedback configs (Cancer -> A or B, positive or negative) ───────────
model_abc_fbA_pos = FeedbackModel(model_if, target_var='A', feedback_var='Cancer', coeff_name='b3', sign=+1.0)
model_abc_fbA_neg = FeedbackModel(model_if, target_var='A', feedback_var='Cancer', coeff_name='b3', sign=-1.0)
model_abc_fbB_pos = FeedbackModel(model_if, target_var='B', feedback_var='Cancer', coeff_name='b3', sign=+1.0)
model_abc_fbB_neg = FeedbackModel(model_if, target_var='B', feedback_var='Cancer', coeff_name='b3', sign=-1.0)
