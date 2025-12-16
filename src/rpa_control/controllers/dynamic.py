"""Dynamic controller with internal state."""
import torch
from .basis import polynomial_basis, get_basis_size, get_basis_names


class DynamicController:
    """Dynamic controller with internal state C.

    Observing parameters: Controller observes state (X, C) to update internal state
        dC/dt = θ_obs · Φ(X, C)

    Actuating parameters: Controller actuates on system using internal state
        u = θ_act · Ψ(C)

    Example:
        dC/dt = o1*A + o2*B² + c1*C  (observing params: o1, o2, c1)
        u = m1*C + m2*C²              (actuating params: m1, m2)
    """

    def __init__(
        self,
        n_state_vars: int,
        n_control_vars: int,
        n_controller_states: int = 1,
        observing_order: int = 2,
        actuating_order: int = 2,
        include_constant: bool = True,
        initial_observing_params: torch.Tensor = None,
        initial_actuating_params: torch.Tensor = None,
        initial_controller_state: torch.Tensor = None,
    ):
        """Initialize dynamic controller.

        Args:
            n_state_vars: Number of ODE state variables
            n_control_vars: Number of control outputs
            n_controller_states: Number of internal controller states
            observing_order: Polynomial order for observing dynamics (dC/dt)
            actuating_order: Polynomial order for actuating control (u)
            include_constant: Whether to include constant term
            initial_observing_params: Initial observing params (n_controller_states x n_basis_obs)
            initial_actuating_params: Initial actuating params (n_control_vars x n_basis_act)
            initial_controller_state: Initial controller state (n_controller_states,)
        """
        self.n_state_vars = n_state_vars
        self.n_control_vars = n_control_vars
        self.n_controller_states = n_controller_states
        self.observing_order = observing_order
        self.actuating_order = actuating_order
        self.include_constant = include_constant

        # Basis size for observing: polynomial in (X, C)
        self.n_basis_observing = get_basis_size(
            n_state_vars + n_controller_states,
            observing_order,
            include_constant
        )

        # Basis size for actuating: polynomial in C only
        self.n_basis_actuating = get_basis_size(
            n_controller_states,
            actuating_order,
            include_constant
        )

        # Initialize observing parameters: shape (n_controller_states, n_basis_observing)
        if initial_observing_params is None:
            self.observing_params = torch.randn(n_controller_states, self.n_basis_observing) * 0.01
            self.observing_params.requires_grad = True
        else:
            assert initial_observing_params.shape == (n_controller_states, self.n_basis_observing)
            self.observing_params = initial_observing_params
            if not self.observing_params.requires_grad:
                self.observing_params.requires_grad = True

        # Initialize actuating parameters: shape (n_control_vars, n_basis_actuating)
        if initial_actuating_params is None:
            self.actuating_params = torch.randn(n_control_vars, self.n_basis_actuating) * 0.01
            self.actuating_params.requires_grad = True
        else:
            assert initial_actuating_params.shape == (n_control_vars, self.n_basis_actuating)
            self.actuating_params = initial_actuating_params
            if not self.actuating_params.requires_grad:
                self.actuating_params.requires_grad = True

        # Initial controller state
        if initial_controller_state is None:
            self.initial_controller_state = torch.zeros(n_controller_states)
        else:
            assert len(initial_controller_state) == n_controller_states
            self.initial_controller_state = initial_controller_state

        # Basis scales for normalization (set during training)
        # Observing: Shape (n_basis_observing,) - RMS of Φ(X, C) over trajectory
        # Actuating: Shape (n_basis_actuating,) - RMS of Ψ(C) over trajectory
        self.observing_basis_scales_per_basis = None
        self.actuating_basis_scales_per_basis = None

    def dynamics(self, state: torch.Tensor, controller_state: torch.Tensor) -> torch.Tensor:
        """Compute controller state derivative: dC/dt = g(X, C).

        Args:
            state: ODE state vector (n_state_vars,)
            controller_state: Controller state vector (n_controller_states,)

        Returns:
            Controller state derivative (n_controller_states,)
        """
        # Concatenate state and controller state
        augmented_state = torch.cat([state, controller_state])

        # Compute basis functions over augmented state
        basis = polynomial_basis(augmented_state, self.observing_order, self.include_constant)

        # Normalize observing basis if scales are available
        if self.observing_basis_scales_per_basis is not None:
            basis = basis / self.observing_basis_scales_per_basis

        # dC/dt = Θ_g · Φ̃(X, C)
        dC_dt = torch.matmul(self.observing_params, basis)

        return dC_dt

    def output(self, controller_state: torch.Tensor) -> torch.Tensor:
        """Compute control output: u = h(C).

        Args:
            controller_state: Controller state vector (n_controller_states,)

        Returns:
            Control vector (n_control_vars,)
        """
        # Compute basis functions over controller state only
        basis = polynomial_basis(controller_state, self.actuating_order, self.include_constant)

        # Normalize actuating basis if scales are available
        if self.actuating_basis_scales_per_basis is not None:
            basis = basis / self.actuating_basis_scales_per_basis

        # u = Θ_h · Ψ̃(C)
        control = torch.matmul(self.actuating_params, basis)

        return control

    def get_dynamics_basis_names(self, state_var_names):
        """Get human-readable names for dynamics basis functions.

        Args:
            state_var_names: Names of ODE state variables

        Returns:
            List of basis function names over (X, C)
        """
        controller_var_names = [f"C{i+1}" for i in range(self.n_controller_states)]
        augmented_names = state_var_names + controller_var_names
        return get_basis_names(augmented_names, self.observing_order, self.include_constant)

    def get_output_basis_names(self):
        """Get human-readable names for output basis functions.

        Returns:
            List of basis function names over C
        """
        controller_var_names = [f"C{i+1}" for i in range(self.n_controller_states)]
        return get_basis_names(controller_var_names, self.actuating_order, self.include_constant)

    def get_param_summary(self, state_var_names, control_names=None, threshold=1e-3):
        """Get human-readable summary of controller parameters.

        Args:
            state_var_names: Names of ODE state variables
            control_names: Names of control variables (default: u1, u2, ...)
            threshold: Parameters below this absolute value are considered zero

        Returns:
            String summary with both full and simplified controller equations
        """
        if control_names is None:
            control_names = [f"u{i+1}" for i in range(self.n_control_vars)]

        controller_state_names = [f"C{i+1}" for i in range(self.n_controller_states)]

        # Convert scaled parameters back to physical parameters for display
        if self.observing_basis_scales_per_basis is not None:
            physical_observing_params = self.observing_params / self.observing_basis_scales_per_basis
        else:
            physical_observing_params = self.observing_params

        if self.actuating_basis_scales_per_basis is not None:
            physical_actuating_params = self.actuating_params / self.actuating_basis_scales_per_basis
        else:
            physical_actuating_params = self.actuating_params

        dynamics_basis_names = self.get_dynamics_basis_names(state_var_names)
        output_basis_names = self.get_output_basis_names()

        lines = []

        # ===== FULL CONTROLLER (all parameters) - show 6 decimals to see small values =====
        lines.append("Full Controller (all parameters):")
        lines.append("  Dynamics:")
        for i, c_name in enumerate(controller_state_names):
            terms = []
            for j, basis_name in enumerate(dynamics_basis_names):
                param_val = physical_observing_params[i, j].item()
                if basis_name == '1':
                    terms.append(f"{param_val:.6f}")
                else:
                    terms.append(f"{param_val:.6f}*{basis_name}")
            lines.append(f"    d{c_name}/dt = {' + '.join(terms)}")

        lines.append("  Output:")
        for i, control_name in enumerate(control_names):
            terms = []
            for j, basis_name in enumerate(output_basis_names):
                param_val = physical_actuating_params[i, j].item()
                if basis_name == '1':
                    terms.append(f"{param_val:.6f}")
                else:
                    terms.append(f"{param_val:.6f}*{basis_name}")
            lines.append(f"    {control_name} = {' + '.join(terms)}")

        # ===== SIMPLIFIED CONTROLLER (only significant parameters) =====
        lines.append(f"\nSimplified Controller (|param| > {threshold}):")
        lines.append("  Dynamics:")
        for i, c_name in enumerate(controller_state_names):
            terms = []
            for j, basis_name in enumerate(dynamics_basis_names):
                param_val = physical_observing_params[i, j].item()
                if abs(param_val) > threshold:
                    if basis_name == '1':
                        terms.append(f"{param_val:.3f}")
                    else:
                        terms.append(f"{param_val:.3f}*{basis_name}")

            if terms:
                lines.append(f"    d{c_name}/dt = {' + '.join(terms)}")
            else:
                lines.append(f"    d{c_name}/dt = 0")

        lines.append("  Output:")
        for i, control_name in enumerate(control_names):
            terms = []
            for j, basis_name in enumerate(output_basis_names):
                param_val = physical_actuating_params[i, j].item()
                if abs(param_val) > threshold:
                    if basis_name == '1':
                        terms.append(f"{param_val:.3f}")
                    else:
                        terms.append(f"{param_val:.3f}*{basis_name}")

            if terms:
                lines.append(f"    {control_name} = {' + '.join(terms)}")
            else:
                lines.append(f"    {control_name} = 0")

        return '\n'.join(lines)
