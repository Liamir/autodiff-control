"""Static controller using polynomial basis functions."""
import torch
from .basis import polynomial_basis, get_basis_size, get_basis_names


class StaticController:
    """Static controller: u = θ_obs·Φ(X)

    Observed parameters: Controller directly observes state X to produce control
        u = θ_obs · Φ(X)

    No internal state or manipulated parameters.

    Example:
        u = o1*A + o2*B²
        where o1, o2 are observed parameters (trainable).
    """

    def __init__(
        self,
        n_state_vars: int,
        n_control_vars: int,
        order: int = 2,
        include_constant: bool = True,
        initial_params: torch.Tensor = None,
    ):
        """Initialize static controller.

        Args:
            n_state_vars: Number of state variables
            n_control_vars: Number of control outputs
            order: Maximum polynomial order for basis functions
            include_constant: Whether to include constant term in basis
            initial_params: Initial parameters (n_control_vars x n_basis)
                          If None, initialized to small random values
        """
        self.n_state_vars = n_state_vars
        self.n_control_vars = n_control_vars
        self.order = order
        self.include_constant = include_constant

        # Calculate basis size
        self.n_basis = get_basis_size(n_state_vars, order, include_constant)

        # Initialize parameters: shape (n_control_vars, n_basis)
        if initial_params is None:
            # Zero initialization for stability (avoids initial explosion)
            self.params = torch.zeros(n_control_vars, self.n_basis, requires_grad=True)
        else:
            assert initial_params.shape == (n_control_vars, self.n_basis), \
                f"Expected shape {(n_control_vars, self.n_basis)}, got {initial_params.shape}"
            self.params = initial_params
            if not self.params.requires_grad:
                self.params.requires_grad = True

    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        """Compute control input from state.

        Args:
            state: State vector of shape (n_state_vars,)

        Returns:
            Control vector of shape (n_control_vars,)
        """
        # Compute basis functions
        basis = polynomial_basis(state, self.order, self.include_constant)

        # Control is linear combination: u = Θ·Φ(X)
        # params: (n_control_vars, n_basis)
        # basis: (n_basis,)
        # output: (n_control_vars,)
        control = torch.matmul(self.params, basis)

        return control

    def get_basis_names(self, var_names):
        """Get human-readable names for basis functions.

        Args:
            var_names: Names of state variables

        Returns:
            List of basis function names
        """
        return get_basis_names(var_names, self.order, self.include_constant)

    def get_param_summary(self, var_names, control_names=None, threshold=1e-3):
        """Get human-readable summary of controller parameters.

        Args:
            var_names: Names of state variables
            control_names: Names of control variables (default: u1, u2, ...)
            threshold: Parameters below this absolute value are considered zero

        Returns:
            String summary of non-zero parameters
        """
        if control_names is None:
            control_names = [f"u{i+1}" for i in range(self.n_control_vars)]

        basis_names = self.get_basis_names(var_names)

        lines = []
        for i, control_name in enumerate(control_names):
            terms = []
            for j, basis_name in enumerate(basis_names):
                param_val = self.params[i, j].item()
                if abs(param_val) > threshold:
                    if basis_name == '1':
                        terms.append(f"{param_val:.3f}")
                    else:
                        terms.append(f"{param_val:.3f}*{basis_name}")

            if terms:
                lines.append(f"{control_name} = {' + '.join(terms)}")
            else:
                lines.append(f"{control_name} = 0")

        return '\n'.join(lines)
