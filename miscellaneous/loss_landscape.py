import torch
from network import Net
import numpy as np
from imageRenderer import Renderer
import torchvision
from pytorch3d.transforms import axis_angle_to_quaternion, matrix_to_axis_angle
import matplotlib.pyplot as plt

batch_size = 100
#This represent the number of true domains
N_domains = 3
N_pixels = 240*240
#This represents the number of domain we think there are
N_input_domains = 4
latent_dim = 1
num_nodes = 1510
K_nearest_neighbors = 30
num_edges = num_nodes*K_nearest_neighbors
dataset_size = 10000
test_set_size = int(dataset_size/10)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_path = "data/vaeContinuousCTFNoisyBiModalAngle/"
pixels_x = np.linspace(-150, 150, num=64).reshape(1, -1)
pixels_y = np.linspace(-150, 150, num=64).reshape(1, -1)
renderer = Renderer(pixels_x, pixels_y, std=1, device=device, use_ctf=True)
training_rotations_matrices = torch.load(dataset_path + "training_rotations_matrices").to(device)
training_images = torch.load(dataset_path + "continuousConformationDataSet")
training_conformation_rotation_matrix = torch.load(dataset_path + "training_conformation_rotation_matrices")
ax_ang = matrix_to_axis_angle(training_conformation_rotation_matrix)
print("Axis angle SHAPE", ax_ang.shape)
print(ax_ang[:, 2, :])
angl = torch.sqrt(torch.sum(ax_ang[:, 2, :]**2, dim=-1))

graph_file="data/features.npy"

features = np.load(graph_file, allow_pickle=True)
features = features.item()
absolute_positions = torch.tensor(features["absolute_positions"] - np.mean(features["absolute_positions"], axis=0))
absolute_positions = absolute_positions.to(device)
local_frame = torch.tensor(features["local_frame"])
local_frame = local_frame.to(device)
net = Net(num_nodes, N_input_domains, latent_dim, None, None, renderer, local_frame,
          absolute_positions, batch_size, device, use_encoder=False)
net.to(device)


mask = torch.zeros(size = (num_nodes, N_input_domains), dtype=torch.float32)
mask[1353:, -1] = 1.0
mask[:1353, 0] = 1.0

axis_rotation = torch.tensor(np.array([[[0, 1, 0], [0, 1, 0], [0, 1, 0], [0,  1,  0]]]), dtype=torch.float32)
angle_rotation = torch.zeros(size=(100, 4), dtype=torch.float32)
##We rotate only the 4th domain to see
angle_rotation[:, 3] = 1.0
ang = torch.tensor(np.linspace(-np.pi, 0, 100), dtype=torch.float32)[:, None]
angle_rotation *= ang

phi = torch.tensor(np.linspace(0, np.pi/2, 100), dtype=torch.float32)[:, None]
z = torch.cos(phi)
y = torch.sin(phi)
axis_rotation = axis_rotation.repeat(100, 1, 1)
axis_rotation[:, -1, :] = torch.concat([torch.zeros((100, 1)), y, z], dim = -1)

print("Axis rot", axis_rotation)
print("ANGLE ROATION", angle_rotation)
net.batch_size = 100
rotation_matrices = training_rotations_matrices[1]
all_new_images = []
#image = training_images[1:2]
image = training_images[4:5]
all_losses = torch.zeros((100, 100), dtype=torch.float32)
for i in range(100):
    print(i)
    angle_rot = angle_rotation[i, -1]
    ax_ang = axis_rotation*angle_rot
    print(axis_rotation)
    ax_ang[:, :-1, 1] = 1


    quaternion = axis_angle_to_quaternion(ax_ang)
    rotation_per_residue = net.compute_rotations(quaternion, mask)
    deformed_structures, _ = net.deform_structure(mask, torch.zeros(size=(100, 4, 3)), rotation_per_residue)
    new_images = renderer.compute_x_y_values_all_atoms(deformed_structures, rotation_matrices)
    all_losses[i, :] = torch.sum((new_images - image) ** 2, dim=(-1, -2))

all_losses = all_losses.detach().cpu().numpy()
print(all_losses.shape)
np.save("all_losses2nd.npy", all_losses)

fig, axs = plt.subplots(2)
all_losses = np.load("all_losses.npy")
all_losses2 = np.load("all_losses2nd.npy")
axs[0].imshow(-all_losses)
axs[0].axhline(y=2/3 * 100)
axs[0].axvline(x=99)
axs[0].set(xlabel="phi", ylabel="rotation angle")
axs[1].imshow(-all_losses2)
#plt.xticks(np.linspace(-np.pi, 0, 100))
#plt.yticks(np.linspace(0, np.pi/2, 100))
axs[1].axhline(y=2/3 * 100)
axs[1].axvline(x=99)
#plt.axhline()
plt.show()
"""
new_images = torch.concat(all_new_images)
#axis_angle = axis_rotation*angle_rotation[:, :, None]
#quaternion = axis_angle_to_quaternion(axis_angle)
#net.batch_size = 100
#rotation_per_residue = net.compute_rotations(quaternion, mask)
#deformed_structures, _ = net.deform_structure(mask, torch.zeros(size=(100, 4, 3)), rotation_per_residue)


image = training_images[9000:9001]
#image_resized = torchvision.transforms.Resize(25)(image)
#print("IMAGE RESIZED SHAPE:", image_resized.shape)
#fourier_image = torch.fft.rfft2(image)
#fourier_image[:, -5:] = 0
#fourier_image[-5:, :] = 0
#fourier_image[:5, :] = 0

#print("Fourier image", fourier_image.shape)
#corrupted_image = torch.fft.irfft2(fourier_image)
#rotation_matrices = training_rotations_matrices[9000]

plt.imshow(training_images[9000], cmap="gray")
plt.show()
#plt.imshow(image_resized[0], cmap="gray")
#plt.show()
#print(image.shape)
#print(rotation_matrices.shape)
#print(deformed_structures.shape)
#new_images = renderer.compute_x_y_values_all_atoms(deformed_structures, rotation_matrices)
#new_images_resize = torchvision.transforms.Resize(25)(new_images)
#print("IMAGES RESIZED SHAPE:", new_images_resize.shape)
#fourier_new_images = torch.fft.rfft2(new_images)
#fourier_new_images[:, :, -5:] = 0
#fourier_new_images[:, -5:, :] = 0
#fourier_new_images[:, :5, :] = 0

#print("Fourier image", fourier_image.shape)
#corrupted_new_images = torch.fft.irfft2(fourier_new_images)
print("New images shape", new_images.shape)

all_losses = []
all_losses = torch.zeros((100, 100), dtype=torch.float32)
for i in range(1000):
    print(i)
    all_losses.append(torch.sum((new_images[i] - image)**2).detach().numpy())
    
"""
"""

all_losses = []
for i in range(10000):
    print(i)
    all_losses.append(torch.sum((new_images_resize[i] - image_resized)**2).detach().numpy())
"""
"""
#all_losses = np.array(all_losses)
#np.save("all_losses.npy", all_losses)

training_rotations_angles = torch.load(dataset_path + "training_rotations_angles").to(device)
print(angl[9000])

plt.plot(ang.detach().numpy(), all_losses)
plt.axvline(x=-2*np.pi/3)
plt.axvline(x=-np.pi/3)
plt.show()
"""





