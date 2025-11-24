"""Wrapper for ODEs with controllers."""
import torch
from rpasim.ode import ODE
from typing import Union, List
from .static import StaticController
from .dynamic import DynamicController
from .basis import polynomial_basis


class ControlledODE(ODE):
    """ODE augmented with a controller.

    For static controller:
        Original ODE: dX/dt = f(X, θ, β)
        With control: dX_i/dt = f_i(X, θ, β) + u_i

    For dynamic controller:
        Augmented state: [X, C]
        dX_i/dt = f_i(X, θ, β) + u_i
        dC/dt = g(X, C)
        u = h(C)

    The controller parameters become the differentiable parameters.
    """

    def __init__(
        self,
        base_ode: ODE,
        controller: Union[StaticController, DynamicController],
        control_indices: List[int],
    ):
        """Initialize controlled ODE.

        Args:
            base_ode: Base ODE system
            controller: Static or dynamic controller
            control_indices: Indices of state variables that receive control input.
                           Must have length equal to controller.n_control_vars.
                           Example: [0, 1] means u[0] is added to dX[0]/dt, u[1] to dX[1]/dt
        """
        self.base_ode = base_ode
        self.controller = controller
        self.control_indices = control_indices

        # Validate control indices
        assert len(control_indices) == controller.n_control_vars, \
            f"Expected {controller.n_control_vars} control indices, got {len(control_indices)}"

        # Get base ODE state dimension
        # Infer from variable_names if state_dim not available
        if hasattr(base_ode, 'state_dim'):
            self.base_state_dim = base_ode.state_dim
        elif hasattr(base_ode, 'variable_names'):
            self.base_state_dim = len(base_ode.variable_names)
        else:
            raise ValueError("Cannot infer state dimension from base ODE")

        # Determine if dynamic controller
        self.is_dynamic = isinstance(controller, DynamicController)

        if self.is_dynamic:
            # State dimension is base + controller states
            state_dim = self.base_state_dim + controller.n_controller_states

            # For dynamic controller, we need to flatten params for the ODE base class
            # but store references to original structured params
            differentiable_params = torch.cat([
                controller.observed_params.flatten(),
                controller.manipulated_params.flatten()
            ]).clone().detach().requires_grad_(True)
        else:
            # State dimension is just base state
            state_dim = self.base_state_dim

            # For static controller, use params directly (already a leaf tensor)
            differentiable_params = controller.params

        # Initialize parent ODE class
        super().__init__(
            differentiable_params=differentiable_params,
            fixed_params=base_ode.fixed_params
        )

        # Store state dimension as attribute
        self.state_dim = state_dim

    def forward(
        self,
        t: torch.Tensor,
        state: torch.Tensor,
        differentiable_params: torch.Tensor = None,
        fixed_params: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute state derivative with controller.

        Args:
            t: Time (scalar)
            state: State vector (state_dim,)
            differentiable_params: Controller parameters (flattened)
            fixed_params: Base ODE fixed parameters

        Returns:
            State derivative (state_dim,)
        """
        if self.is_dynamic:
            # Split state into base and controller parts
            base_state = state[:self.base_state_dim]
            controller_state = state[self.base_state_dim:]

            # Reshape controller params from flat vector
            dynamics_size = self.controller.observed_params.numel()
            observed_params = differentiable_params[:dynamics_size].reshape(self.controller.observed_params.shape)
            manipulated_params = differentiable_params[dynamics_size:].reshape(self.controller.manipulated_params.shape)

            # Compute base ODE dynamics
            base_derivative = self.base_ode(t, base_state, None, fixed_params)

            # Compute controller output (manipulated): u = θ_man · Ψ(C)
            output_basis = polynomial_basis(controller_state, self.controller.manipulated_order, self.controller.include_constant)
            control = torch.matmul(manipulated_params, output_basis)

            # Compute controller dynamics (observed): dC/dt = θ_obs · Φ(X, C)
            augmented_state = torch.cat([base_state, controller_state])
            dynamics_basis = polynomial_basis(augmented_state, self.controller.observed_order, self.controller.include_constant)
            controller_derivative = torch.matmul(observed_params, dynamics_basis)

            # Add control to specified state variables
            for i, control_idx in enumerate(self.control_indices):
                base_derivative[control_idx] = base_derivative[control_idx] + control[i]

            # Concatenate derivatives
            derivative = torch.cat([base_derivative, controller_derivative])

        else:
            # Static controller
            base_state = state

            # differentiable_params is already (n_control_vars, n_basis) shape - no reshape needed

            # Compute base ODE dynamics
            base_derivative = self.base_ode(t, base_state, None, fixed_params)

            # Compute control (observed): u = θ_obs · Φ(X)
            basis = polynomial_basis(base_state, self.controller.order, self.controller.include_constant)
            control = torch.matmul(differentiable_params, basis)

            # Add control to specified state variables
            for i, control_idx in enumerate(self.control_indices):
                base_derivative[control_idx] = base_derivative[control_idx] + control[i]

            derivative = base_derivative

        return derivative

    def get_initial_state(self, base_initial_state: torch.Tensor) -> torch.Tensor:
        """Get initial state for controlled ODE.

        Args:
            base_initial_state: Initial state for base ODE

        Returns:
            Initial state including controller state if dynamic
        """
        if self.is_dynamic:
            # Concatenate base state with controller initial state
            return torch.cat([base_initial_state, self.controller.initial_controller_state])
        else:
            return base_initial_state

    def extract_base_state(self, state: torch.Tensor) -> torch.Tensor:
        """Extract base ODE state from augmented state.

        Args:
            state: Full state vector (possibly including controller state)

        Returns:
            Base ODE state only
        """
        return state[:self.base_state_dim]

    def update_controller_params(self):
        """Update controller parameters from differentiable_params.

        This syncs the flat parameter vector back to controller structure.
        Needed after optimization steps.
        """
        if self.is_dynamic:
            # Split differentiable_params back into dynamics and output params
            dynamics_size = self.controller.observed_params.numel()
            output_size = self.controller.manipulated_params.numel()

            dynamics_flat = self.differentiable_params[:dynamics_size]
            output_flat = self.differentiable_params[dynamics_size:dynamics_size + output_size]

            # Reshape and update (maintaining gradient connection)
            self.controller.observed_params = dynamics_flat.reshape(self.controller.observed_params.shape)
            self.controller.manipulated_params = output_flat.reshape(self.controller.manipulated_params.shape)
        else:
            # Reshape flat params back to controller params
            self.controller.params = self.differentiable_params.reshape(self.controller.params.shape)

    def get_controller_summary(self, state_var_names, control_names=None, threshold=1e-3):
        """Get human-readable summary of controller.

        Args:
            state_var_names: Names of base ODE state variables
            control_names: Names of control variables
            threshold: Parameters below this absolute value are considered zero

        Returns:
            String summary of controller
        """
        # Update controller params first to ensure they're in sync
        self.update_controller_params()

        return self.controller.get_param_summary(state_var_names, control_names, threshold)

    def __str__(self) -> str:
        """Return string representation."""
        controller_type = "Dynamic" if self.is_dynamic else "Static"
        base_name = self.base_ode.name if hasattr(self.base_ode, 'name') else str(type(self.base_ode).__name__)
        return f"Controlled ODE: {base_name} + {controller_type} Controller"
