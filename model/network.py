import torch

class Net(torch.nn.Module):
    def __init__(self, N_residues, N_domains, latent_dim, encoder, decoder, renderer, local_frame, device, use_encoder = True):
        super(Net, self).__init__()
        self.N_residues = N_residues
        self.N_domains = N_domains
        self.latent_dim = latent_dim
        self.epsilon_mask_loss = 1e-10
        self.encoder = encoder
        self.decoder = decoder
        self.renderer = renderer
        self.tau = 0.05
        ##Next line compute the coordinates in the local frame (N_atoms,3)
        self.device = device
        self.latent_mean = torch.nn.Parameter(data=torch.randn((100000, self.latent_dim), device=device), requires_grad=True)
        self.latent_std = torch.ones((100000, self.latent_dim), device=device)*0.01
        self.prior_std = self.latent_std
        self.use_encoder = use_encoder
        self.residues = torch.arange(0, self.N_residues, 1, dtype=torch.float32, device=device)[:, None]

        self.cluster_means_mean = torch.nn.Parameter(data=torch.tensor([160, 550, 800, 1300], dtype=torch.float32,device=device)[None, :],
                                                requires_grad=True)

        self.cluster_means_std = torch.nn.Parameter(data=torch.tensor([10, 10, 10, 10], dtype=torch.float32, device=device)[None, :],
                                              requires_grad=True)

        self.cluster_std_mean = torch.nn.Parameter(data=torch.tensor([100, 100, 100, 100], dtype=torch.float32, device=device)[None, :],
                                              requires_grad=True)

        self.cluster_std_std = torch.nn.Parameter(data=torch.tensor([10, 10, 10, 10], dtype=torch.float32, device=device)[None, :],
                                              requires_grad=True)

        self.cluster_proportions_mean = torch.nn.Parameter(torch.zeros(4, dtype=torch.float32, device=device)[None, :],
                                                      requires_grad=True)

        self.cluster_proportions_std = torch.nn.Parameter(torch.ones(4, dtype=torch.float32, device=device)[None, :],
                           requires_grad=True)


        self.cluster_parameters = {"means":{"mean":self.cluster_means_mean, "std":self.cluster_means_std},
                                   "stds":{"mean":self.cluster_std_mean, "std":self.cluster_std_std},
                                   "proportions":{"mean":self.cluster_proportions_mean, "std":self.cluster_proportions_std}}


        self.cluster_prior_means_mean = torch.tensor([160, 550, 800, 1300], dtype=torch.float32,device=device)[None, :]
        self.cluster_prior_means_std = torch.tensor([100, 100, 100, 100], dtype=torch.float32, device=device)[None, :]
        self.cluster_prior_std_mean = torch.tensor([100, 100, 100, 100], dtype=torch.float32, device=device)[None, :]
        self.cluster_prior_std_std = torch.tensor([10, 10, 10, 10], dtype=torch.float32, device=device)[None, :]
        self.cluster_prior_proportions_mean = torch.zeros(4, dtype=torch.float32, device=device)[None, :]
        self.cluster_prior_proportions_std = torch.ones(4, dtype=torch.float32, device=device)[None, :]

        self.cluster_prior = {"means":{"mean":self.cluster_prior_means_mean, "std": self.cluster_prior_means_std},
                                   "stds":{"mean":self.cluster_prior_std_mean, "std": self.cluster_prior_std_std},
                                   "proportions":{"mean":self.cluster_prior_proportions_mean,"std":self.cluster_prior_proportions_std}}


    def compute_mask(self):
        cluster_proportions = torch.randn(4, device=self.device)*self.cluster_proportions_std + self.cluster_proportions_mean
        cluster_means = torch.randn(4, device=self.device)*self.cluster_means_std + self.cluster_means_mean
        cluster_std = torch.randn(4, device=self.device)*self.cluster_std_std + self.cluster_std_mean
        proportions = torch.softmax(cluster_proportions, dim=1)
        log_num = -0.5*(self.residues - cluster_means)**2/cluster_std**2 + \
              torch.log(proportions)

        mask = torch.softmax(log_num/self.tau, dim=1)
        return mask

    def encode(self, images):
        """
        Encode images into latent varaibles
        :param images: (N_batch, N_pix_x, N_pix_y) containing the cryoEM images
        :return: (N_batch, 2*N_domains) predicted gaussian distribution over the latent variables
        """
        flattened_images = torch.flatten(images, start_dim=1, end_dim=2)
        distrib_parameters = self.encoder.forward(flattened_images)
        return distrib_parameters

    def reparameterize(self, distrib_parameters, indexes=None):
        """
        Sample from the approximate posterior over the latent space
        :param distrib_parameters: (N_batch, 2*latent_dim) the parameters mu, sigma of the approximate posterior
        :return: (N_batch, latent_dim) actual samples.
        """
        batch_size = distrib_parameters.shape[0]
        if self.use_encoder:
            latent_mean = distrib_parameters[:, :self.latent_dim]
            latent_std = distrib_parameters[:, self.latent_dim:]
            latent_vars = latent_std*torch.randn(size=(batch_size, self.latent_dim), device=self.device) + latent_mean
            return latent_vars, latent_mean, latent_std

        latent_vars = self.latent_std[indexes]*torch.randn(size=(batch_size, self.latent_dim), device=self.device)\
                      + self.latent_mean[indexes]

        return latent_vars

    def forward(self, indexes, images=None):
        """
        Encode then decode image
        :images: (N_batch, N_pix_x, N_pix_y) cryoEM images
        :return: tensors: a new structure (N_batch, N_residues, 3), the attention mask (N_residues, N_domains),
                translation vectors for each residue (N_batch, N_residues, 3) leading to the new structure.
        """
        flattened_images = torch.flatten(images, start_dim=1, end_dim=2)
        distrib_parameters = self.encoder.forward(flattened_images)
        weights = self.compute_mask()
        if self.use_encoder:
            latent_variables, latent_mean, latent_std = self.reparameterize(distrib_parameters, None)
        else:
            latent_variables = self.reparameterize(None, indexes)

        output = self.decoder.forward(latent_variables)
        if self.use_encoder:
            return output, weights, latent_variables, latent_mean, latent_std
        else:
            return output, weights, latent_variables, None, None




