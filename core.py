import math
import random
import torch
from torch import nn
import torch.nn.functional as F

class QubitNormalization(nn.Module):
    def __init__(self, num_wires, temp, noise_strength=0.001, dropout_prob=0.01, apply_noise=False):
        super(QubitNormalization, self).__init__()
        self.num_wires = num_wires
        self.temp = temp
        self.noise_strength = noise_strength
        self.dropout_prob = dropout_prob
        self.apply_noise = apply_noise

    def forward(self, x):
        bs, in_size = x.shape
        x = F.softmax(x / self.temp.to(x.device), dim=1).sqrt()
        pad_size = (2 ** self.num_wires) - in_size
        if pad_size > 0:
            x = F.pad(x, (0, pad_size))
        if self.apply_noise:
            noise = torch.randn_like(x) * self.noise_strength
            x = x + noise
            keep_prob = 1 - self.dropout_prob
            dropout_mask = torch.bernoulli(torch.full_like(x, keep_prob)) / keep_prob
            x = x * dropout_mask
            x = x / torch.sqrt(torch.sum(torch.abs(x) ** 2, dim=-1, keepdim=True))
        return x

class Unitary(nn.Module):
    def __init__(self, num_filters, num_layers, num_wires, device, apply_noise=False):
        super(Unitary, self).__init__()
        self.num_filters = num_filters
        self.num_layers = num_layers
        self.num_wires = num_wires
        self.device = device
        self.pi_half = math.pi / 2
        self.apply_noise = apply_noise

    def forward(self, weight, phi):
        weight = (weight + torch.randn_like(weight) * 0.001) if self.apply_noise else weight
        final_product = torch.eye(2**self.num_wires, dtype=torch.complex64, device=self.device).unsqueeze(0).repeat(self.num_filters, 1, 1)
        for layer in range(self.num_layers):
            current_weight = weight[:, layer, :, :]
            angle = current_weight[..., 0] * self.pi_half
            cos_term = torch.cos(angle)
            sin_term = torch.sin(angle)
            exp_term1 = torch.exp(1j * phi[:, layer, :])
            exp_term2 = torch.exp(1j * current_weight[..., 1])

            U = torch.zeros(self.num_filters, self.num_wires, 2, 2, dtype=torch.complex64, device=self.device)
            U[..., 0, 0] = cos_term
            U[..., 0, 1] = -sin_term * exp_term2
            U[..., 1, 0] = sin_term * exp_term1
            U[..., 1, 1] = cos_term * exp_term1 * exp_term2

            kron_product = U[:, 0]
            for i in range(1, self.num_wires):
                kron_product = torch.einsum('bij,bkl->bikjl', kron_product, U[:, i]).view(self.num_filters, 2**(i+1), 2**(i+1))

            final_product = torch.bmm(final_product, kron_product)
        return final_product

class Measurement(nn.Module):
    def __init__(self, num_wires, device):
        super(Measurement, self).__init__()
        self.num_wires = num_wires
        self.device = device
        self.obs_list = self._prepare_measurements()

    def _prepare_measurements(self):
        I = torch.eye(2, dtype=torch.complex64, device=self.device)
        Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=self.device)
        results = []
        for i in range(self.num_wires):
            operator = torch.tensor(1, dtype=torch.complex64, device=self.device)
            for j in range(self.num_wires):
                if i == j:
                    operator = torch.kron(operator, Z)
                else:
                    operator = torch.kron(operator, I)
            results.append(operator)
        return torch.stack(results).to(self.device)

    def forward(self):
        return self.obs_list

class Quanv1d(nn.Module):
"""
Our proposed Quanv1D layer that works on data of shape (batch_size, channels, sequence_length).

Parameters
----------
temp : float
    Regulates the distribution of attention across time points while classical-to-quantum encoding. Higher values result in more evenly 
    distributed attention, while lower values emphasize discrete feature selection.
    
shots : int
    Number of repetitions for quantum measurement to gather statistics. If 0, analytical values are used.

noise : bool
    If True, enables quantum noise simulation to emulate real-world quantum behavior.
"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, dilation: int,
                 bias=False, device="cpu", temp: float=1.0, shots: int=0, noise=False):
        super().__init__()
        self.num_wires = torch.ceil(torch.log2(torch.max(torch.tensor([torch.tensor(in_channels * kernel_size, dtype=torch.float32), 2])))).int().item()
        self.num_filters = (out_channels + self.num_wires - 1) // self.num_wires
        self.num_layers = kernel_size
        self.out_channels = out_channels
        self.patcher = nn.Unfold(kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(padding, 0), dilation=(dilation, 1))
        self.device = device
        self.temp = torch.tensor(temp, dtype=torch.float32)
        self.shots = shots
        self.noise = noise

        self.qubit_norm = QubitNormalization(self.num_wires, self.temp, apply_noise=self.noise)
        self.unitary_op = Unitary(self.num_filters, self.num_layers, self.num_wires, self.device, apply_noise=self.noise)
        self.measurement_op = Measurement(self.num_wires, self.device)

        self.weight = nn.Parameter(nn.init.kaiming_uniform_(torch.randn((self.num_filters, self.num_layers, self.num_wires, 2), device=self.device)), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(out_channels, device=self.device), requires_grad=True) if bias else None
        self.phi = torch.rand((self.num_filters, self.num_layers, self.num_wires), device=self.device, requires_grad=False)

    def apply_depolarizing_noise(self, x):
        if torch.rand(1).item() >= 0.01:
            return x
        else:
            I = torch.eye(2, dtype=torch.complex64, device=self.device)
            X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=self.device)
            Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device=self.device)
            Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=self.device)
            paulis = [X, Y, Z]

            gate = random.choice(paulis)
            target_wire = random.randint(0, self.num_wires - 1)

            U = []
            for i in range(self.num_wires):
                U.append(gate if i == target_wire else I)
            full_unitary = U[0]
            for i in range(1, self.num_wires):
                full_unitary = torch.kron(full_unitary, U[i])

            return torch.einsum('bfij,jk->bfik', x, full_unitary)

    def adjust_probabilities_with_shots(self, x):
        p = torch.abs(x) ** 2

        noise_scale = torch.sqrt(p * (1 - p) / self.shots)
        x = x + torch.complex(
            torch.randn_like(x.real) * noise_scale,
            torch.randn_like(x.imag) * noise_scale
        )

        x = x / torch.sqrt(torch.sum(torch.abs(x) ** 2, dim=-1, keepdim=True))
        return x

    def forward(self, x):
        x = torch.unsqueeze(x, -1)
        x = self.patcher(x)
        bs, ch_ker, l = x.shape

        x = x.permute(0, 2, 1).contiguous().view(bs * l, ch_ker)
        x = self.qubit_norm(x).to(torch.cfloat)

        unitaries = self.unitary_op(self.weight, self.phi)
        x = torch.einsum('bik,fjk->bfij', x.unsqueeze(dim=1), unitaries)

        x = self.apply_depolarizing_noise(x) if self.noise else x

        x = self.adjust_probabilities_with_shots(x) if self.shots>0 else x

        intermediate = torch.einsum('bfij,wjk->bfwik', x.conj(), self.measurement_op())
        x = torch.einsum('bfwik,bfik->bfw', intermediate, x)

        x = x.real.contiguous().view(bs * l, -1)

        if self.bias is not None:
            x = x[:, :self.out_channels] + self.bias
        else:
            x = x[:, :self.out_channels]

        x = x.contiguous().view(bs, l, self.out_channels).permute(0, 2, 1)
        return x

class FQN(nn.Module):
    def __init__(self, num_channels: int, num_classes: int, device,
                 dim: int=16, depth: int=3, input_window: int=15, input_scale: int=2, hidden_window: int=5):
        super().__init__()
        self.input = Quanv1d(in_channels=num_channels, out_channels=dim, kernel_size=input_window, padding=(input_window-1)//2,
                             stride=input_scale, dilation=1, device=device)

        def quanv_bn_relu(in_channels, out_channels, hidden_window, dilation):
            return nn.Sequential(
                Quanv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=hidden_window, padding=0,
                        stride=1, dilation=dilation, device=device),
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
            )

        self.hidden_layers = nn.Sequential(
            *[quanv_bn_relu(dim, dim, hidden_window, i+1) for i in range(depth)]
        )

        self.output = Quanv1d(in_channels=dim, out_channels=num_classes, kernel_size=1, padding=0, stride=1, dilation=1,
                              device=device)

    def forward(self, x):
        x = self.input(x)
        x = self.hidden_layers(x)
        x = self.output(x)
        x = F.adaptive_avg_pool1d(x, 1)
        return x.view(x.size(0), -1)
