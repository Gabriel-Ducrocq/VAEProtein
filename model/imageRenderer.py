import torch
#import torch_geometric
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt



class Renderer():
    def __init__(self, pixels_x, pixels_y, N_heavy=4530, period= 300/128, std = 1, defocus= 5000, spherical_aberration=21,
                 accelerating_voltage=300 , amplitude_contrast_ratio = 0.06, device="cpu", use_ctf=True):
        self.std_blob = std
        self.len_x = pixels_x.shape[1]
        self.len_y = pixels_y.shape[1]
        assert self.len_x == self.len_y, "Number of pixels different on x and y"
        assert self.len_x % 2 == 0, "Number of pixel is not a multiple of 2"
        self.pixels_x = torch.tensor(pixels_x, dtype=torch.float32, device=device)
        self.pixels_y = torch.tensor(pixels_y, dtype=torch.float32, device=device)
        self.grid_x = self.pixels_x.repeat(self.len_y, 1)
        self.grid_y = torch.transpose(self.pixels_y, dim0=-2, dim1=-1).repeat(1, self.len_x)
        self.grid = torch.concat([self.grid_x[:, :, None], self.grid_y[:, :, None]], dim=2)
        self.N_heavy_atoms = N_heavy
        self.torch_sqrt_2pi= torch.sqrt(torch.tensor(2*np.pi, device=device))
        self.defocus = defocus
        self.spherical_aberration = spherical_aberration
        self.accelerating_voltage = accelerating_voltage # see the paper cited by cryoSparc site on CTF.
        self.amplitude_contrast_ratio = amplitude_contrast_ratio
        self.grid_period = period
        self.device = device
        self.use_ctf = use_ctf
        self.frequencies = torch.tensor([k/(period*self.len_x) for k in range(-int(self.len_x/2), int(self.len_x/2))],
                                        device=device)

        freqs = self.frequencies[:, None]**2 + self.frequencies[None, -int(self.len_y/2 + 1):]**2
        self.ctf_grid = self.compute_ctf_np(freqs, accelerating_voltage, spherical_aberration, amplitude_contrast_ratio,
                                            defocus)

    def compute_ctf_np(self,
            freqs: np.ndarray,
            volt: float,
            cs: float,
            w: float,
            df: float,
            phase_shift: float = 0,
            bfactor = None,
    ) -> np.ndarray:
        """
        Compute the 2D CTF
        Input:
            freqs (np.ndarray) Nx2 array of 2D spatial frequencies
            dfu (float): DefocusU (Angstrom)
            dfv (float): DefocusV (Angstrom)
            dfang (float): DefocusAngle (degrees)
            volt (float): accelerating voltage (kV)
            cs (float): spherical aberration (Å)
            w (float): amplitude contrast ratio
            phase_shift (float): degrees
            bfactor (float): envelope fcn B-factor (Angstrom^2)
        """
        # convert units
        volt = volt * 1000
        phase_shift = phase_shift * np.pi / 180

        # lam = sqrt(h^2/(2*m*e*Vr)); Vr = V + (e/(2*m*c^2))*V^2
        lam = 12.2639 / np.sqrt(volt + 0.97845e-6 * volt ** 2)
        s2 = freqs
        gamma = (
                2 * np.pi * (-0.5 * df * lam * s2 + 0.25 * cs * lam ** 3 * s2 ** 2)
                - phase_shift
        )

        ctf = torch.sqrt(torch.tensor(1 - w ** 2, device=self.device)) * torch.sin(torch.tensor(gamma, device=self.device))\
                         - w * torch.cos(torch.tensor(gamma, device=self.device))
        if bfactor is not None:
            ctf *= np.exp(-bfactor / 4 * s2)

        return ctf

    def compute_gaussian_kernel(self, x, pixels_pos):
        """
        Computes the values of the gaussian kernel for one axis only but all heavy atoms and samples in batch
        :param x: (N_batch, 1): the coordinate of all heavy atoms on one axis for all samples in batch.
        :return: (N_batch, N_atoms, N_pix)
        """
        batch_size = x.shape[0]
        scaled_distances = -(1/2)*(torch.broadcast_to(pixels_pos, (batch_size, self.N_heavy_atoms, -1)) -
                                   x[:, :, None])**2/self.std_blob**2
        axis_val = torch.exp(scaled_distances)/self.torch_sqrt_2pi
        return axis_val

    def ctf_corrupting(self, image):
        """
        Corrupts the image with the CTF.
        :param image: torch tensor (N_batch, N_pixels_s, N_pixels_y), non corrupted image.
        :return:  torch tensor (N_batch, N_pixels_s, N_pixels_y), corrupted image
        """
        fourier_images = torch.fft.rfft2(image)
        corrupted_fourier = fourier_images*self.ctf_grid
        corrupted_images = torch.fft.irfft2(corrupted_fourier)
        return corrupted_images

    def compute_x_y_values_all_atoms(self, atom_positions, rotation_matrices):
        """

        :param atom_position: (N_batch, N_atoms, 3)
        :rotation_matrices: (N_batch, 3, 3)
        :return:
        """
        transposed_atom_positions = torch.transpose(atom_positions, dim0=1, dim1=2)
        rotated_transposed_atom_positions = torch.matmul(rotation_matrices, transposed_atom_positions)
        rotated_atom_positions = torch.transpose(rotated_transposed_atom_positions, 1, 2)
        #projected_densities = self.compute_gaussian_kernel(rotated_atom_positions[:, :, :2])
        all_x = self.compute_gaussian_kernel(rotated_atom_positions[:, :, 0], self.pixels_x)
        all_y = self.compute_gaussian_kernel(rotated_atom_positions[:, :, 1], self.pixels_y)
        prod = torch.einsum("bki,bkj->bkij", (all_x, all_y))
        projected_densities = torch.sum(prod, dim=1)
        if self.use_ctf:
            projected_densities = self.ctf_corrupting(projected_densities)

        return projected_densities

"""
pixels_x = np.linspace(-150, 150, num = 128).reshape(1, -1)
pixels_y = np.linspace(-150, 150, num = 128).reshape(1, -1)
rend = Renderer(pixels_x, pixels_y)
k = torch.tensor(np.linspace(0, 1, 1000))
ctf = rend.ctf_grid
plt.imshow(ctf[:, :])
plt.show()

file = "data/features.npy"
device = "cpu"
features = np.load(file, allow_pickle=True)
features = features.item()
absolute_positions = features["absolute_positions"]
absolute_positions = absolute_positions - np.mean(absolute_positions, axis=0)
absolute_positions = absolute_positions.reshape(1, -1, 3)
absolute_positions = torch.tensor(absolute_positions).to(device)

pixels_x = np.linspace(-150, 150, num = 64).reshape(1, -1)
pixels_y = np.linspace(-150, 150, num = 64).reshape(1, -1)
rend = Renderer(pixels_x, pixels_y, std=1)
k = torch.tensor(np.linspace(0, 0.5, 1000))
print(torch.min(absolute_positions[0, :, 0]))
print(torch.max(absolute_positions[0, :, 0]))

print(torch.min(absolute_positions[0, :, 1]))
print(torch.max(absolute_positions[0, :, 1]))

res = rend.compute_x_y_values_all_atoms(absolute_positions, torch.eye(3)[None, :, :])
res = res[0].detach().numpy()
print("power:", np.var(res))
res += np.random.normal(scale=0.01, size=(64, 64))

print(np.unique(res))
print(res.shape)

plt.imshow(res.T, cmap="gray")
plt.show()
"""


