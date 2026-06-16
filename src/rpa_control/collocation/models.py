"""CasADi ODE models for direct collocation optimal control."""
import casadi as ca
import numpy as np
from abc import ABC, abstractmethod


class CasadiODE(ABC):
    """Base class for CasADi ODE models."""

    name: str = "ODE"
    n_states: int = 0
    n_controls: int = 0
    n_ext_inputs: int = 0
    state_names: list[str] = []
    control_names: list[str] = []

    @abstractmethod
    def dynamics(self, x: ca.MX, u: ca.MX, p: ca.MX | None = None) -> ca.MX:
        """Compute dx/dt as CasADi MX expression.

        Args:
            x: State vector (n_states,)
            u: Control vector (n_controls,)
            p: External inputs (n_ext_inputs,), e.g. stressor. None if unused.

        Returns:
            xdot: State derivative (n_states,)
        """
        raise NotImplementedError


class PopulationDynamicsCasadi(CasadiODE):
    """Lotka-Volterra predator-prey dynamics.

    dx1/dt = a*x1 - b*x1*x2
    dx2/dt = -c*x2 + d*x1*x2 + u
    """

    name = "Population Dynamics (Lotka-Volterra)"
    n_states = 2
    n_controls = 1
    n_ext_inputs = 0
    state_names = ["prey", "predator"]
    control_names = ["u"]

    def __init__(self, a=0.5, b=0.025, c=0.5, d=0.005):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def dynamics(self, x, u, p=None):
        x1, x2 = x[0], x[1]
        dx1 = self.a * x1 - self.b * x1 * x2
        dx2 = -self.c * x2 + self.d * x1 * x2 + u[0]
        return ca.vertcat(dx1, dx2)


class MelanomaCasadi(CasadiODE):
    """Dimensionless UV homeostasis / melanoma model.

    dD/dt  = p1*UV/m - p2*D
    dm/dt  = p3*M*D - p4*m
    dα/dt  = p8*(D - 1)
    dM/dt  = M*(p5*u + p6*α*D + p7*D - 1)

    Control u: multiplicative factor on p5 (pM). u=1 → no treatment.
    """

    name = "Melanoma (UV Homeostasis)"
    n_states = 4
    n_controls = 1
    n_ext_inputs = 0
    state_names = ["D", "m", "alpha", "M"]
    control_names = ["u_pM"]

    def __init__(self, p1=364.0, p2=364.0, p3=13.0, p4=13.0,
                 p5=0.1, p6=0.05, p7=0.05, p8=None, uv=3.0):
        import math
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        self.p5 = p5
        self.p6 = p6
        self.p7 = p7
        self.p8 = p8 if p8 is not None else 1.0 / (5.0 * math.log(2.0))
        self.uv = uv

    def dynamics(self, x, u, p=None):
        D, m, alpha, M = x[0], x[1], x[2], x[3]
        u_pM = u[0]

        dD = self.p1 * self.uv / m - self.p2 * D
        dm = self.p3 * M * D - self.p4 * m
        da = self.p8 * (D - 1.0)
        dM = M * (self.p5 * u_pM + self.p6 * alpha * D + self.p7 * D - 1.0)

        return ca.vertcat(dD, dm, da, dM)


class HPACasadi(CasadiODE):
    """HPA axis model with 9 control inputs and stressor external input.

    States: [CRH, ACTH, Cortisol, Pituitary, Adrenal]
    Controls: [I1, I2, I3, C1, C2, C3, A1, A2, A3]
    External input: p[0] = stressor u(t)
    """

    name = "HPA Axis (Stress Response)"
    n_states = 5
    n_controls = 9
    n_ext_inputs = 1
    state_names = ["CRH", "ACTH", "Cortisol", "Pituitary", "Adrenal"]
    control_names = ["I1", "I2", "I3", "C1", "C2", "C3", "A1", "A2", "A3"]

    def __init__(self, params=None):
        if params is None:
            params = {
                'gamma_x1': (np.log(2) / 4) * 24 * 60,    # ~249.5 /day
                'gamma_x2': (np.log(2) / 20) * 24 * 60,   # ~49.9 /day
                'gamma_x3': (np.log(2) / 80) * 24 * 60,   # ~12.5 /day
                'gamma_P': np.log(2) / 20,                  # ~0.035 /day
                'gamma_A': np.log(2) / 30,                  # ~0.023 /day
                'KGR': 4.0,
                'nGR': 3.0,
                'KP': 1e6,
                'KA': 1e6,
            }
        self.params = params

    def dynamics(self, x, u, p=None):
        x1, x2, x3, P, A = x[0], x[1], x[2], x[3], x[4]
        I1, I2, I3 = u[0], u[1], u[2]
        C1, C2, C3 = u[3], u[4], u[5]
        A1, A2, A3 = u[6], u[7], u[8]

        stressor = p[0] if p is not None else 1.0

        prm = self.params
        eps = 1e-8
        x3_eff = C3 * x3

        # Receptor functions
        MR = 1.0 / (x3_eff + eps)
        GR = 1.0 / ((x3_eff / prm['KGR']) ** prm['nGR'] + 1.0)

        dx1 = prm['gamma_x1'] * (I1 * stressor * MR * GR - A1 * x1)
        dx2 = prm['gamma_x2'] * (I2 * (C1 * x1) * P * GR - A2 * x2)
        dx3 = prm['gamma_x3'] * (I3 * (C2 * x2) * A - A3 * x3)
        dP = prm['gamma_P'] * P * ((C1 * x1) * (1.0 - P / prm['KP']) - 1.0)
        dA = prm['gamma_A'] * A * ((C2 * x2) * (1.0 - A / prm['KA']) - 1.0)

        return ca.vertcat(dx1, dx2, dx3, dP, dA)
