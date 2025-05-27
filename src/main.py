"""
PuzzleTensor: A Method-Agnostic Data Transformation for Compact Tensor Factorization
"""

import math
import time
import random
import numpy as np
import tensorly as tl
import torch
from torch import nn, optim
from tensorly.decomposition import parafac, tucker, matrix_product_state
from tensorly.cp_tensor import cp_to_tensor


class PuzzleTensor(nn.Module):
    """
    PuzzleTensor module that applies multiple layers of frequency-domain shifts
    to the input tensor in order to transform it into a form that yields a lower
    reconstruction error when factorized.

    Attributes:
        batch (int): The size of the batch (first dimension of input tensor).
        shape (tuple): The remaining dimensions of the tensor.
        D (int): Number of dimensions (excluding the batch dimension).
        num_layer (int): The number of frequency shifting layers to apply.
        S (nn.ModuleList): A list of parameter lists for each layer containing learnable shift factors.
        phase (list): List of frequency phase factors for full Fourier components.
        phase_half (list): List of frequency phase factors for half Fourier components.
        dim (list): For each layer, the tuple of dimensions along which FFT/IFFT is applied.
        s (list): For each layer, the shape (tuple) used in the inverse FFT.
        perm (list): Permutation order used in the loss calculation.
        coshape (list): Flattened shape for each mode in the loss calculation.
        norm_factor (list): Normalization factor for each mode in the loss calculation.
    """

    def __init__(self, shape, init, num_layer):
        """
        Initialize the PuzzleTensor module.

        Parameters:
            shape (tuple): A tuple specifying the shape of the input tensor,
                           where the first element is the batch size and the
                           rest are the spatial (or other) dimensions.
            init (str): Initialization type for the learnable parameters.
                        'r' for random initialization, 'z' for zero initialization.
            num_layer (int): The number of frequency shifting layers to apply.
        """
        super().__init__()
        self.batch = shape[0]
        self.shape = shape[1:]
        self.D = len(self.shape)
        self.num_layer = num_layer

        if init == 'r':
            init_func = torch.rand
        elif init == 'z':
            init_func = torch.zeros
        else:
            raise ValueError(f'Unknown initialization type: {init}')

        # Create a ModuleList to hold the shift parameter lists for each layer
        self.S = nn.ModuleList()
        for layer in range(num_layer):
            view_shape = (self.batch,) + tuple(-1 if i == (layer % self.D) else 1 for i in range(self.D))
            params = [nn.Parameter(init_func([self.batch, self.shape[layer % self.D]]).view(view_shape))
                      for _ in range(self.D - 1)]
            self.S.append(nn.ParameterList(params))

        # Precompute phase factors for FFT shifting
        self.phase, self.phase_half = [], []
        for d in range(self.D - 1):
            view_shape = tuple(-1 if i == d else 1 for i in range(self.D))
            sym_arange = (torch.arange(self.shape[d]) + self.shape[d] // 2) % self.shape[d] - self.shape[d] // 2
            sym_arange[self.shape[d] // 2] *= self.shape[d] % 2
            self.phase.append(2.0j * torch.pi * sym_arange.view(view_shape) / self.shape[d])
        for d in range(self.D - 2, self.D):
            view_shape = tuple(-1 if i == d else 1 for i in range(self.D))
            asym_arange = torch.arange(self.shape[d] // 2 + 1)
            asym_arange[-1] *= self.shape[d] % 2
            self.phase_half.append(2.0j * torch.pi * asym_arange.view(view_shape) / self.shape[d])

        # Precompute auxiliary values
        self.dim, self.s = [], []
        for layer in range(self.num_layer):
            # Determine the dimensions (axes) to apply FFT/IFFT (skip the current shifting axis)
            self.dim.append(tuple(i + 1 for i in range(self.D) if i != (layer % self.D)))
            # Calculate the corresponding sizes for inverse FFT
            self.s.append(tuple(sh for i, sh in enumerate(self.shape) if i != (layer % self.D)))
        self.perm, self.coshape, self.norm_factor = [], [], []
        for k in range(self.D):
            self.perm.append([0, k + 1] + [i + 1 for i in range(self.D) if i != k])
            self.coshape.append(math.prod(self.shape) // self.shape[k])
            self.norm_factor.append(1.0 / math.sqrt(self.shape[k]))

    def forward(self, x, reverse=False):
        """
        Forward pass of the PuzzleTensor module.

        This method applies a sequence of frequency-domain transformations
        (phase shifts) across multiple layers. The order of these layers
        depends on whether 'reverse' is set:
            - If reverse=False (default), layers are processed in ascending
              order (0 to num_layer-1). In the very first layer (layer=0), we
              assume 'x' is already in frequency domain and skip the FFT.
            - If reverse=True, layers are processed in descending order
              (num_layer-1 down to 0), and each step begins with an FFT
              (because we assume 'x' is now in the spatial domain from the end
              of the forward pass).

        For each layer, the following steps occur:
          1. FFT along dimensions self.dim[layer], unless this is
             the first layer in the forward (reverse=False) pass.
          2. Compute the exponential phase factors from the learnable shift
             parameters in self.S[layer].
          3. Multiply x by these phase factors in the frequency domain
             (or their conjugates if reverse=True).
          4. Apply an inverse FFT (IFFT) to transform back to the spatial
             domain along the same dimensions.

        Parameters:
            x (torch.Tensor): The input tensor to be transformed. If reverse=False
                              and layer=0, x is expected to be in the frequency domain.
                              Otherwise, x should be in the spatial domain if reverse=True.
            reverse (bool, optional): Whether to apply the transformations
                                      in reverse order (default: False).

        Returns:
            torch.Tensor: The tensor after applying all the frequency-domain shifts
                          and inverse transforms.
        """
        for layer in range(self.num_layer) if not reverse else reversed(range(self.num_layer)):
            x = torch.fft.rfftn(x, dim=self.dim[layer]) if layer or reverse else x
            ph = torch.exp(self.S[layer][self.D - 2] * self.phase_half[(layer % self.D) <= (self.D - 2)])
            for k in range(self.D - 2):
                ph = ph * torch.exp(self.S[layer][k] * self.phase[(layer % self.D <= k) + k])
            x = torch.fft.irfftn((ph if not reverse else torch.conj(ph)) * x, s=self.s[layer], dim=self.dim[layer])
        return x

    def loss(self, x):
        """
        Computes the sum of nuclear norms of the tensor after it is reshaped
        along different modes.

        Parameters:
            x (torch.Tensor): A tensor on which the nuclear norm along various
                              modes is computed.

        Returns:
            torch.Tensor: Scalar loss value (sum of nuclear norms, each scaled
                          by the corresponding normalization factor).
        """
        loss = torch.tensor(0.)
        for k in range(self.D):
            x_perm = x.permute(self.perm[k]).contiguous()
            x_k = x_perm.view(self.batch, self.shape[k], self.coshape[k])
            norm_k = torch.linalg.matrix_norm(x_k, ord='nuc')
            loss += torch.sum(norm_k) * self.norm_factor[k]
        return loss


def split_tensor(tensor, block_shape):
    """
    Splits an arbitrary n-dimensional tensor into smaller blocks along each axis
    as specified by block_shape, and returns an (n+1)-dimensional tensor stacking these blocks.

    Parameters:
        tensor (torch.Tensor): n-dimensional input tensor with shape [D_0, D_1, ..., D_{n-1}].
        block_shape (list or tuple of ints): Specifies the number of blocks to split each axis into.
            Must have the same length as the number of dimensions of the tensor, and each D_i
            must be divisible by block_shape[i].

    Returns:
        torch.Tensor: A stacked (n+1)-dimensional tensor containing the blocks.
                      The shape is [B_0 * B_1 * ... * B_{n-1}, D_0/B_0, D_1/B_1, ..., D_{n-1}/B_{n-1}],
                      where B_i = block_shape[i].
    """
    if tensor.dim() != len(block_shape):
        raise ValueError("Length of block_shape must match the number of tensor dimensions.")

    blocks = [tensor]
    for axis, num_chunks in enumerate(block_shape):
        new_blocks = []
        for block in blocks:
            new_blocks.extend(torch.chunk(block, num_chunks, dim=axis))
        blocks = new_blocks

    return torch.stack(blocks)


def train(x, shape, epochs=6001, init='z', num_layer=None, verbose=1):
    """
    Trains the PuzzleTensor model on the provided tensor 'x' by optimizing
    its learnable shift parameters. This function employs a staged training
    approach where certain layers' parameters have their gradients disabled
    until specific epochs are reached.

    Parameters:
        x (torch.Tensor): Input tensor (preferably in the frequency domain) on
                          which to train the model.
        shape (tuple): Shape of the input tensor including the batch dimension.
        epochs (int, optional): Total number of training epochs (default: 6001).
        init (str, optional): Initialization type for the PuzzleTensor
                              ('r' for random, 'z' for zeros; default: 'z').
        num_layer (int, optional): Number of frequency shifting layers in the
                                   PuzzleTensor. If None, user must set it externally.
        verbose (int, optional): Verbose

    Returns:
        PuzzleTensor: A trained PuzzleTensor model containing the learned shift
                      parameters for each layer.

    """
    model = PuzzleTensor(shape, init=init, num_layer=num_layer)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    start = time.time()
    for epoch in range(epochs):
        z = model(x)
        loss = model.loss(z)

        # Staged Training
        if epoch == 0:
            for d in range(model.D - 1):
                model.S[-3][d].requires_grad = False
                model.S[-2][d].requires_grad = False
                model.S[-1][d].requires_grad = False
        elif epoch == epochs // 4:
            for d in range(model.D - 1):
                model.S[-3][d].requires_grad = True
        elif epoch == 2 * epochs // 4:
            for d in range(model.D - 1):
                model.S[-2][d].requires_grad = True
        elif epoch == 3 * epochs // 4:
            for d in range(model.D - 1):
                model.S[-1][d].requires_grad = True
        else:
            pass

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose and epoch % 100 == 0:
            print(f'epoch: {epoch}\t|\tloss: {loss:.4f} \t|\ttime: {time.time() - start:.4f}')
            start = time.time()

    return model


def main():
    SEED = 42
    np.set_printoptions(precision=7, linewidth=900, suppress=True)
    torch.set_printoptions(precision=7, sci_mode=False, linewidth=900)
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    # Synthetic Data
    I, J, K = 32, 32, 32
    mask = torch.rand(I, J, K) > 0.99
    x = torch.zeros(I, J, K)
    x[mask] = 1

    # Hyperparameters
    INIT = 'z'
    EPOCHS = 8001
    NUM_LAYER = 9
    block_shape = [1, 1, 1]
    ranks_CP_1 = [27, 67, 133]
    ranks_TK_1 = [11, 17, 22]
    ranks_TT_1 = [8, 13, 19]
    ranks_CP_2 = [21, 61, 127]
    ranks_TK_2 = [9, 16, 22]
    ranks_TT_2 = [7, 12, 19]

    # Preprocessing: Split the tensor into blocks
    D = x.dim()
    x = split_tensor(x, block_shape=block_shape)
    # Apply FFT to convert the data to the frequency domain
    fx = torch.fft.rfftn(x, dim=tuple(i for i in range(1 - D, 0)))
    # Train PuzzleTensor to learn frequency shifts
    model = train(fx, shape=x.shape, epochs=EPOCHS, init=INIT, num_layer=NUM_LAYER)
    fx = fx.detach()
    # Apply the learned shifts to solve the puzzle
    z = model(fx)

    # (Optional) Measure the distortion induced by shifting
    pred = model(z, reverse=True)
    error = torch.norm(x - pred) / torch.norm(x)
    print(f"Error induced by shifting: {error:.7f}")

    # Set Tensorly backend to PyTorch
    tl.set_backend('pytorch')

    # Evaluate CP decomposition on both tensors
    errors_CP_1, errors_CP_2 = [], []
    for r in ranks_CP_1:
        temp_diff, temp_norm = [], []
        for b in range(x.shape[0]):
            weights, factors = parafac(x[b], rank=r, normalize_factors=False, tol=1e-2)
            reconstructed_tensor = cp_to_tensor((weights, factors))
            temp_diff.append(torch.norm(x[b] - reconstructed_tensor).item())
            temp_norm.append(torch.norm(x[b]).item())
        errors_CP_1.append(math.hypot(*temp_diff) / math.hypot(*temp_norm))
    for r in ranks_CP_2:
        temp_diff, temp_norm = [], []
        for b in range(z.shape[0]):
            weights, factors = parafac(z[b], rank=r, normalize_factors=False, tol=1e-2)
            reconstructed_tensor = cp_to_tensor((weights, factors))
            temp_diff.append(torch.norm(z[b] - reconstructed_tensor).item())
            temp_norm.append(torch.norm(z[b]).item())
        errors_CP_2.append(math.hypot(*temp_diff) / math.hypot(*temp_norm))

    # Evaluate Tucker (TK) decomposition on both tensors
    errors_TK_1, errors_TK_2 = [], []
    for r in ranks_TK_1:
        temp_diff, temp_norm = [], []
        for b in range(x.shape[0]):
            core, factors = tucker(x[b], rank=[int(r)] * D)
            reconstructed_tensor = tl.tucker_to_tensor((core, factors))
            temp_diff.append(torch.norm(x[b] - reconstructed_tensor).item())
            temp_norm.append(torch.norm(x[b]).item())
        errors_TK_1.append(math.hypot(*temp_diff) / math.hypot(*temp_norm))
    for r in ranks_TK_2:
        temp_diff, temp_norm = [], []
        for b in range(z.shape[0]):
            core, factors = tucker(z[b], rank=[int(r)] * D)
            reconstructed_tensor = tl.tucker_to_tensor((core, factors))
            temp_diff.append(torch.norm(z[b] - reconstructed_tensor).item())
            temp_norm.append(torch.norm(z[b]).item())
        errors_TK_2.append(math.hypot(*temp_diff) / math.hypot(*temp_norm))

    # Evaluate Tensor-Train (TT) decomposition on both tensors
    errors_TT_1, errors_TT_2 = [], []
    for r in ranks_TT_1:
        temp_diff, temp_norm = [], []
        for b in range(x.shape[0]):
            tt_factors = matrix_product_state(x[b], rank=int(r))
            reconstructed_tensor = tl.tt_to_tensor(tt_factors)
            temp_diff.append(torch.norm(x[b] - reconstructed_tensor).item())
            temp_norm.append(torch.norm(x[b]).item())
        errors_TT_1.append(math.hypot(*temp_diff) / math.hypot(*temp_norm))
    for r in ranks_TT_2:
        temp_diff, temp_norm = [], []
        for b in range(z.shape[0]):
            tt_factors = matrix_product_state(z[b], rank=int(r))
            reconstructed_tensor = tl.tt_to_tensor(tt_factors)
            temp_diff.append(torch.norm(z[b] - reconstructed_tensor).item())
            temp_norm.append(torch.norm(z[b]).item())
        errors_TT_2.append(math.hypot(*temp_diff) / math.hypot(*temp_norm))

    print("\nReconstruction Errors")
    print(f"CP                :", "\t".join(map(lambda a: f"{a:8.4f}", errors_CP_1)))
    print(f"CP + PuzzleTensor :", "\t".join(map(lambda a: f"{a:8.4f}", errors_CP_2)))
    print(f"TK                :", "\t".join(map(lambda a: f"{a:8.4f}", errors_TK_1)))
    print(f"TK + PuzzleTensor :", "\t".join(map(lambda a: f"{a:8.4f}", errors_TK_2)))
    print(f"TT                :", "\t".join(map(lambda a: f"{a:8.4f}", errors_TT_1)))
    print(f"TT + PuzzleTensor :", "\t".join(map(lambda a: f"{a:8.4f}", errors_TT_2)))


if __name__ == '__main__':
    main()
