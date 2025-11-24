"""Dynamic controller with internal state."""
import torch
from .basis import polynomial_basis, get_basis_size, get_basis_names


class DynamicController:
    """Dynamic controller with internal state C.

    Observed parameters: Controller observes state (X, C) to update internal state
        dC/dt = θ_obs · Φ(X, C)

    Manipulated parameters: Controller manipulates system using internal state
        u = θ_man · Ψ(C)

    Example:
        dC/dt = o1*A + o2*B² + c1*C  (observed params: o1, o2, c1)
        u = m1*C + m2*C²              (manipulated params: m1, m2)
    """

    def __init__(
        self,
        n_state_vars: int,
        n_control_vars: int,
        n_controller_states: int = 1,
        observed_order: int = 2,
        manipulated_order: int = 2,
        include_constant: bool = True,
        initial_observed_params: torch.Tensor = None,
        initial_manipulated_params: torch.Tensor = None,
        initial_controller_state: torch.Tensor = None,
    ):
        """Initialize dynamic controller.

        Args:
            n_state_vars: Number of ODE state variables
            n_control_vars: Number of control outputs
            n_controller_states: Number of internal controller states
            observed_order: Polynomial order for observed dynamics (dC/dt)
            manipulated_order: Polynomial order for manipulated control (u)
            include_constant: Whether to include constant term
            initial_observed_params: Initial observed params (n_controller_states x n_basis_obs)
            initial_manipulated_params: Initial manipulated params (n_control_vars x n_basis_man)
            initial_controller_state: Initial controller state (n_controller_states,)
        """
        self.n_state_vars = n_state_vars
        self.n_control_vars = n_control_vars
        self.n_controller_states = n_controller_states
        self.observed_order = observed_order
        self.manipulated_order = manipulated_order
        self.include_constant = include_constant

        # Basis size for observed: polynomial in (X, C)
        self.n_basis_observed = get_basis_size(
            n_state_vars + n_controller_states,
            observed_order,
            include_constant
        )

        # Basis size for manipulated: polynomial in C only
        self.n_basis_manipulated = get_basis_size(
            n_controller_states,
            manipulated_order,
            include_constant
        )

        # Initialize observed parameters: shape (n_controller_states, n_basis_observed)
        if initial_observed_params is None:
            self.observed_params = torch.randn(n_controller_states, self.n_basis_observed) * 0.01
            self.observed_params.requires_grad = True
        else:
            assert initial_observed_params.shape == (n_controller_states, self.n_basis_observed)
            self.observed_params = initial_observed_params
            if not self.observed_params.requires_grad:
                self.observed_params.requires_grad = True

        # Initialize manipulated parameters: shape (n_control_vars, n_basis_manipulated)
        if initial_manipulated_params is None:
            self.manipulated_params = torch.randn(n_control_vars, self.n_basis_manipulated) * 0.01
            self.manipulated_params.requires_grad = True
        else:
            assert initial_manipulated_params.shape == (n_control_vars, self.n_basis_manipulated)
            self.manipulated_params = initial_manipulated_params
            if not self.manipulated_params.requires_grad:
                self.manipulated_params.requires_grad = True

        # Initial controller state
        if initial_controller_state is None:
            self.initial_controller_state = torch.zeros(n_controller_states)
        else:
            assert len(initial_controller_state) == n_controller_states
            self.initial_controller_state = initial_controller_state

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
        basis = polynomial_basis(augmented_state, self.observed_order, self.include_constant)

        # dC/dt = Θ_g · Φ(X, C)
        dC_dt = torch.matmul(self.observed_params, basis)

        return dC_dt

    def output(self, controller_state: torch.Tensor) -> torch.Tensor:
        """Compute control output: u = h(C).

        Args:
            controller_state: Controller state vector (n_controller_states,)

        Returns:
            Control vector (n_control_vars,)
        """
        # Compute basis functions over controller state only
        basis = polynomial_basis(controller_state, self.manipulated_order, self.include_constant)

        # u = Θ_h · Ψ(C)
        control = torch.matmul(self.manipulated_params, basis)

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
        return get_basis_names(augmented_names, self.observed_order, self.include_constant)

    def get_output_basis_names(self):
        """Get human-readable names for output basis functions.

        Returns:
            List of basis function names over C
        """
        controller_var_names = [f"C{i+1}" for i in range(self.n_controller_states)]
        return get_basis_names(controller_var_names, self.manipulated_order, self.include_constant)

    def get_param_summary(self, state_var_names, control_names=None, threshold=1e-3):
        """Get human-readable summary of controller parameters.

        Args:
            state_var_names: Names of ODE state variables
            control_names: Names of control variables (default: u1, u2, ...)
            threshold: Parameters below this absolute value are considered zero

        Returns:
            String summary of controller dynamics and output
        """
        if control_names is None:
            control_names = [f"u{i+1}" for i in range(self.n_control_vars)]

        controller_state_names = [f"C{i+1}" for i in range(self.n_controller_states)]

        # Get dynamics summary
        dynamics_basis_names = self.get_dynamics_basis_names(state_var_names)
        lines = ["Controller Dynamics:"]

        for i, c_name in enumerate(controller_state_names):
            terms = []
            for j, basis_name in enumerate(dynamics_basis_names):
                param_val = self.observed_params[i, j].item()
                if abs(param_val) > threshold:
                    if basis_name == '1':
                        terms.append(f"{param_val:.3f}")
                    else:
                        terms.append(f"{param_val:.3f}*{basis_name}")

            if terms:
                lines.append(f"  d{c_name}/dt = {' + '.join(terms)}")
            else:
                lines.append(f"  d{c_name}/dt = 0")

        # Get output summary
        output_basis_names = self.get_output_basis_names()
        lines.append("\nControl Output:")

        for i, control_name in enumerate(control_names):
            terms = []
            for j, basis_name in enumerate(output_basis_names):
                param_val = self.manipulated_params[i, j].item()
                if abs(param_val) > threshold:
                    if basis_name == '1':
                        terms.append(f"{param_val:.3f}")
                    else:
                        terms.append(f"{param_val:.3f}*{basis_name}")

            if terms:
                lines.append(f"  {control_name} = {' + '.join(terms)}")
            else:
                lines.append(f"  {control_name} = 0")

        return '\n'.join(lines)
