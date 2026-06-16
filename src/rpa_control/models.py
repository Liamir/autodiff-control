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

def make_cancer_f(oidx, c1=c1_cancer, c2=c2_cancer):
    """Return extra_f for Cancer given the output variable index oidx."""
    def cancer_f(t, y, O_star):
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

model_if = RPAModel(
    name='IF',
    state_vars=['A', 'B'],
    output_var='B',
    params_base=dict(mu=1.0, alpha=10.0, beta1=10.0, beta2=1.0),
    equations=[
        alpha * (mu_sym - B),
        beta1 * A_sym - beta2 * B,
    ],
    extra_vars=['Cancer'], extra_f=make_cancer_f(1),
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
