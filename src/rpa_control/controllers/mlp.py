"""MLP (Multi-Layer Perceptron) controller."""
import torch


class MLPController:
    """MLP controller: u = MLP(X)

    Single hidden layer neural network that maps state directly to control.

    Architecture:
        hidden = activation(W1 @ X + b1)
        u = W2 @ hidden + b2

    Parameters are stored as a flat vector and unpacked during forward pass.
    """

    def __init__(
        self,
        n_state_vars: int,
        n_control_vars: int,
        n_hidden: int = 8,
        activation: str = 'tanh',
        initial_params: torch.Tensor = None,
    ):
        """Initialize MLP controller.

        Args:
            n_state_vars: Number of state variables (input dimension)
            n_control_vars: Number of control outputs (output dimension)
            n_hidden: Number of hidden layer neurons (default: 8)
            activation: Activation function - 'tanh', 'relu', or 'sigmoid' (default: 'tanh')
            initial_params: Initial parameters as flat vector (if None, use Xavier init)
        """
        self.n_state_vars = n_state_vars
        self.n_control_vars = n_control_vars
        self.n_hidden = n_hidden
        self.activation = activation

        # Calculate parameter dimensions
        # W1: (n_hidden, n_state_vars), b1: (n_hidden,)
        # W2: (n_control_vars, n_hidden), b2: (n_control_vars,)
        self.n_params_W1 = n_hidden * n_state_vars
        self.n_params_b1 = n_hidden
        self.n_params_W2 = n_control_vars * n_hidden
        self.n_params_b2 = n_control_vars
        self.n_params_total = self.n_params_W1 + self.n_params_b1 + self.n_params_W2 + self.n_params_b2

        # Initialize parameters as flat vector
        if initial_params is None:
            self.params = self._xavier_init()
        else:
            assert initial_params.shape == (self.n_params_total,), \
                f"Expected shape ({self.n_params_total},), got {initial_params.shape}"
            self.params = initial_params
            if not self.params.requires_grad:
                self.params.requires_grad = True

    def _xavier_init(self):
        """Xavier/Glorot initialization for stable training."""
        # W1: fan_in=n_state_vars, fan_out=n_hidden
        std_W1 = (2.0 / (self.n_state_vars + self.n_hidden)) ** 0.5
        W1 = torch.randn(self.n_params_W1) * std_W1

        # b1: zeros
        b1 = torch.zeros(self.n_params_b1)

        # W2: fan_in=n_hidden, fan_out=n_control_vars
        std_W2 = (2.0 / (self.n_hidden + self.n_control_vars)) ** 0.5
        W2 = torch.randn(self.n_params_W2) * std_W2

        # b2: small values (not zero, for asymmetry breaking)
        b2 = torch.randn(self.n_params_b2) * 0.01

        params = torch.cat([W1, b1, W2, b2])
        params.requires_grad = True
        return params

    def _unpack_params(self, params=None):
        """Unpack flat parameter vector into W1, b1, W2, b2.

        Args:
            params: Flat parameter vector (if None, use self.params)

        Returns:
            (W1, b1, W2, b2) tuple of tensors
        """
        if params is None:
            params = self.params

        idx = 0
        W1 = params[idx:idx + self.n_params_W1].reshape(self.n_hidden, self.n_state_vars)
        idx += self.n_params_W1

        b1 = params[idx:idx + self.n_params_b1]
        idx += self.n_params_b1

        W2 = params[idx:idx + self.n_params_W2].reshape(self.n_control_vars, self.n_hidden)
        idx += self.n_params_W2

        b2 = params[idx:idx + self.n_params_b2]

        return W1, b1, W2, b2

    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        """Compute control input from state using MLP forward pass.

        Args:
            state: State vector of shape (n_state_vars,)

        Returns:
            Control vector of shape (n_control_vars,)
        """
        W1, b1, W2, b2 = self._unpack_params()

        # Hidden layer: h = activation(W1 @ state + b1)
        hidden = torch.matmul(W1, state) + b1

        if self.activation == 'tanh':
            hidden = torch.tanh(hidden)
        elif self.activation == 'relu':
            hidden = torch.relu(hidden)
        elif self.activation == 'sigmoid':
            hidden = torch.sigmoid(hidden)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

        # Output layer: u = W2 @ hidden + b2
        control = torch.matmul(W2, hidden) + b2

        return control

    def get_param_summary(self, var_names, control_names=None, threshold=1e-3):
        """Get human-readable summary of MLP architecture and parameters.

        Args:
            var_names: Names of state variables
            control_names: Names of control variables (default: u1, u2, ...)
            threshold: Not used for MLP (kept for interface compatibility)

        Returns:
            String summary of MLP architecture
        """
        if control_names is None:
            control_names = [f"u{i+1}" for i in range(self.n_control_vars)]

        W1, b1, W2, b2 = self._unpack_params()

        lines = []
        lines.append("=" * 60)
        lines.append("MLP Controller Architecture")
        lines.append("=" * 60)
        lines.append(f"Input layer:   {self.n_state_vars} neurons ({', '.join(var_names)})")
        lines.append(f"Hidden layer:  {self.n_hidden} neurons (activation: {self.activation})")
        lines.append(f"Output layer:  {self.n_control_vars} neurons ({', '.join(control_names)})")
        lines.append(f"Total parameters: {self.n_params_total}")
        lines.append("")
        lines.append("Parameter shapes:")
        lines.append(f"  W1: ({self.n_hidden}, {self.n_state_vars}) - input to hidden weights")
        lines.append(f"  b1: ({self.n_hidden},) - hidden biases")
        lines.append(f"  W2: ({self.n_control_vars}, {self.n_hidden}) - hidden to output weights")
        lines.append(f"  b2: ({self.n_control_vars},) - output biases")
        lines.append("")
        lines.append("Parameter statistics:")
        lines.append(f"  W1: mean={W1.mean().item():.4f}, std={W1.std().item():.4f}, range=[{W1.min().item():.4f}, {W1.max().item():.4f}]")
        lines.append(f"  b1: mean={b1.mean().item():.4f}, std={b1.std().item():.4f}, range=[{b1.min().item():.4f}, {b1.max().item():.4f}]")
        lines.append(f"  W2: mean={W2.mean().item():.4f}, std={W2.std().item():.4f}, range=[{W2.min().item():.4f}, {W2.max().item():.4f}]")
        lines.append(f"  b2: mean={b2.mean().item():.4f}, std={b2.std().item():.4f}, range=[{b2.min().item():.4f}, {b2.max().item():.4f}]")
        lines.append("=" * 60)

        return '\n'.join(lines)
