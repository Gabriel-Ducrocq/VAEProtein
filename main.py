import model.utils as utils
from model.mlp import MLP
from model.network import Net
import numpy as np
import torch
import torch.optim.lr_scheduler
from torch.utils.data import DataLoader
import time
from model.imageRenderer import Renderer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_ACCUMULATION_STEP = 1

batch_size = 100
#This represent the number of true domains
N_domains = 3
N_pixels = 64*64
#This represents the number of domain we think there are
N_input_domains = 4
latent_dim = 1
N_residues = 1510
K_nearest_neighbors = 30
dataset_size = 10000
test_set_size = int(dataset_size/10)

print("Is cuda available ?", torch.cuda.is_available())

def train_loop(network, absolute_positions, renderer, local_frame,
               dataset_path):
    optimizer = torch.optim.Adam(network.parameters(), lr=0.0003)
    atom_relative_positions = torch.matmul(absolute_positions, local_frame)
    all_losses = []
    all_rmsd = []
    all_dkl_losses = []
    all_tau = []
    local_frame_in_rows = torch.transpose(local_frame, 0, 1)
    local_frame_in_columns = local_frame

    all_cluster_means_loss = []
    all_cluster_std_loss = []
    all_cluster_proportions_loss = []
    all_lr = []

    training_rotations_matrices = torch.load(dataset_path + "training_rotations_matrices").to(device)
    training_images = torch.load(dataset_path + "continuousConformationDataSet")
    training_indexes = torch.tensor(np.array(range(10000)))
    for epoch in range(0,5000):
        epoch_loss = torch.empty(100)
        data_loader = iter(DataLoader(training_indexes, batch_size=batch_size, shuffle=True))
        for idx, batch_indexes in enumerate(data_loader):
            start = time.time()
            print("epoch:", epoch)
            print("--Batch percentage:", idx/100)
            deformed_images = training_images[batch_indexes]
            batch_rotation_matrices = training_rotations_matrices[batch_indexes]
            deformed_images = deformed_images.to(device)
            batch_indexes = batch_indexes.to(device)
            transforms, mask, latent_variables, latent_mean, latent_std = network.forward(batch_indexes, deformed_images)
            new_structures = utils.process_structure(transforms, atom_relative_positions,mask, local_frame_in_rows,
                                                     local_frame_in_columns, device, network)

            new_images = renderer.compute_x_y_values_all_atoms(new_structures, batch_rotation_matrices)

            loss, rmsd, Dkl_loss, Dkl_mask_mean, Dkl_mask_std, Dkl_mask_proportions = utils.loss(network, new_images,
                                                    mask, deformed_images, None, latent_mean=latent_mean, latent_std=latent_std)
            loss = loss/NUM_ACCUMULATION_STEP
            loss.backward()
            if ((idx + 1) % NUM_ACCUMULATION_STEP == 0) or (idx + 1 == len(data_loader)):
                # Update Optimizer
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss[idx] = loss.cpu().detach()
            print("Printing metrics")
            all_losses.append(loss.cpu().detach())
            all_dkl_losses.append(Dkl_loss.cpu().detach())
            all_rmsd.append(rmsd.cpu().detach())
            all_tau.append(network.tau)
            all_cluster_means_loss.append(Dkl_mask_mean.cpu().detach())
            all_cluster_std_loss.append(Dkl_mask_std.cpu().detach())
            all_cluster_proportions_loss.append(Dkl_mask_proportions.cpu().detach())
            end = time.time()
            print("Running time one iteration:", end-start)
            print("\n\n")

        print("\n\n\n\n")
        np.save(dataset_path + "losses_train.npy", np.array(all_losses))
        np.save(dataset_path +"losses_dkl.npy", np.array(all_dkl_losses))
        np.save(dataset_path +"losses_rmsd.npy", np.array(all_rmsd))
        np.save(dataset_path + "losses_cluster_mean", np.array(all_cluster_means_loss))
        np.save(dataset_path + "losses_cluster_std", np.array(all_cluster_std_loss))
        np.save(dataset_path + "losses_cluster_proportions", np.array(all_cluster_proportions_loss))
        np.save(dataset_path +"all_tau.npy", np.array(all_tau))
        np.save(dataset_path + "all_lr.npy", np.array(all_lr))
        mask = network.compute_mask()
        mask_python = mask.to("cpu").detach()
        np.save(dataset_path +"mask"+str(epoch)+".npy", mask_python)
        torch.save(network.state_dict(), dataset_path +"model")
        torch.save(network, dataset_path +"full_model")

def experiment(dataset_path, local_frame_file="data/features.npy"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = np.load(local_frame_file, allow_pickle=True)
    features = features.item()
    absolute_positions = torch.tensor(features["absolute_positions"] - np.mean(features["absolute_positions"], axis=0))
    absolute_positions = absolute_positions.to(device)
    local_frame = torch.tensor(features["local_frame"])
    local_frame = local_frame.to(device)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder_mlp = MLP(latent_dim, 2*3*N_input_domains, 350, device, num_hidden_layers=2)
    encoder_mlp = MLP(N_pixels, latent_dim*2, [2048, 1024, 512, 512], device, num_hidden_layers=4)

    pixels_x = np.linspace(-150, 150, num=64).reshape(1, -1)
    pixels_y = np.linspace(-150, 150, num=64).reshape(1, -1)
    renderer = Renderer(pixels_x, pixels_y, std=1, device=device, use_ctf=True)

    net = Net(N_residues, N_input_domains, latent_dim, encoder_mlp, decoder_mlp, renderer, local_frame, device, use_encoder=True)
    net.to(device)
    train_loop(net, absolute_positions, renderer, local_frame, dataset_path=dataset_path)


if __name__ == '__main__':
    print("Is cuda available ?", torch.cuda.is_available())
    experiment("../VAEProtein/data/vaeContinuousCTFNoisyBiModalAngle10kEncoderOldFashioned/")
