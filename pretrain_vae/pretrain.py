import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from utils import min_distance_sqr_pbc, radius_graph_pbc, get_pbc_distances, repeat_blocks
from utils import frac_to_cart_coords, cart_to_frac_coords
from torch_scatter import scatter
from utils import KHOT_EMBEDDINGS, MAX_ATOMIC_NUM
from config import cfg
from pretrain_vae.gemnet import GemNetT as Decoder
from pretrain_vae.formula import FormulaNet
from pretrain_vae.dimenet import DimeNetPlusPlusWrap as DimeNet



def build_mlp(in_dim, hidden_dim, fc_num_layers, out_dim):
    mods = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for i in range(fc_num_layers - 1):
        mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    mods += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*mods)


class crysVAE(nn.Module):
    def __init__(self):
        super().__init__()
        '''
        1. self.encoder input : formula graph output: latent z
        2. reparametrize
        3. predict lattice, num atoms, composition
        4. self.decoder input: latent, predicted lattice output: edges
        '''
        self.encoder = DimeNet()
        self.decoder = Decoder()
        self.fc_mu = nn.Linear(cfg.VAE.hidden_dim,
                               cfg.VAE.hidden_dim)
        self.fc_var = nn.Linear(cfg.VAE.hidden_dim,
                                cfg.VAE.hidden_dim)

        self.fc_num_atoms = build_mlp(cfg.VAE.hidden_dim, cfg.VAE.hidden_dim, 1, cfg.MAX_ATOMS+1)
        self.fc_lattice = build_mlp(cfg.VAE.hidden_dim, cfg.VAE.hidden_dim, 1, 6)
        self.fc_composition = build_mlp(cfg.VAE.hidden_dim, cfg.VAE.hidden_dim, 1, cfg.MAX_ATOMIC_NUM)
        self.lattice_scaler = torch.load(cfg.SCALER_DIR + "lattice_scaler.pt")
        self.fc_property = build_mlp(cfg.VAE.hidden_dim, cfg.VAE.hidden_dim, 1,1)
        self.device = cfg.DEVICE
        self.fc_atom = nn.Linear(cfg.DECODER.latent_dim, cfg.MAX_ATOMIC_NUM) #gemnet laten dim 512
        sigmas = torch.tensor(np.exp(np.linspace(
            np.log(cfg.VAE.sigma_begin),
            np.log(cfg.VAE.sigma_end),
            cfg.VAE.num_noise_level)), dtype=torch.float32)

        self.sigmas = nn.Parameter(sigmas, requires_grad=False)

        type_sigmas = torch.tensor(np.exp(np.linspace(
            np.log(cfg.VAE.type_sigma_begin),
            np.log(cfg.VAE.type_sigma_end),
            cfg.VAE.num_noise_level)), dtype=torch.float32)

        self.type_sigmas = nn.Parameter(type_sigmas, requires_grad=False)

        self.embedding = torch.zeros(100, 92)
        for i in range(100):
            self.embedding[i] = torch.tensor(KHOT_EMBEDDINGS[i + 1])

        # obtain from datamodule.
        # self.scaler = None
    def encode(self, batch):
        hidden = self.encoder(batch)
        mu = self.fc_mu(hidden)
        log_var = self.fc_var(hidden)
        z = self.reparameterize(mu, log_var)
        return mu, log_var, z

    def decode_stats(self, z, gt_num_atoms, gt_lengths, gt_angles, teacher_forcing):
        """
        decode key stats from latent embeddings.
        batch is input during training for teach-forcing
        """
        num_atoms = self.predict_num_atoms(z)
        lengths_and_angles, lengths, angles = (
            self.predict_lattice(z, gt_num_atoms))
        composition_per_atom = self.predict_composition(z, gt_num_atoms)
        if teacher_forcing:
            lengths = gt_lengths
            angles = gt_angles
        return num_atoms, lengths_and_angles, lengths, angles, composition_per_atom

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def predict_num_atoms(self, z):
        return self.fc_num_atoms(z)

    def predict_property(self, z):
        self.scaler.match_device(z)
        return self.scaler.inverse_transform(self.fc_property(z))

    def predict_lattice(self, z, num_atoms):
        self.lattice_scaler.match_device(z)
        pred_lengths_and_angles = self.fc_lattice(z)  # (N, 6)
        scaled_preds = self.lattice_scaler.inverse_transform(
            pred_lengths_and_angles)
        pred_lengths = scaled_preds[:, :3]
        pred_angles = scaled_preds[:, 3:]
        pred_lengths = pred_lengths * num_atoms.view(-1, 1).float() ** (1 / 3)
        # <pred_lengths_and_angles> is scaled.
        return pred_lengths_and_angles, pred_lengths, pred_angles

    def predict_composition(self, z, num_atoms):
        z_per_atom = z.repeat_interleave(num_atoms, dim=0)
        pred_composition_per_atom = self.fc_composition(z_per_atom)
        return pred_composition_per_atom


    def kld_loss(self, mu, log_var):
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        return kld_loss

    def forward(self, batch):
        mu, log_var, z = self.encode(batch)
        (pred_num_atoms, pred_lengths_and_angles, pred_lengths, pred_angles,
         pred_composition_per_atom) = self.decode_stats(
            z, batch.num_atoms_t, batch.lengths_t, batch.angles_t, teacher_forcing = True)

        noise_level = torch.randint(0, self.sigmas.size(0),
                                    (batch.num_atoms_t.size(0),),
                                    device=self.device)
        used_sigmas_per_atom = self.sigmas[noise_level].repeat_interleave(
            batch.num_atoms_t, dim=0)

        type_noise_level = torch.randint(0, self.type_sigmas.size(0),
                                         (batch.num_atoms_t.size(0),),
                                         device=self.device)
        used_type_sigmas_per_atom = (
            self.type_sigmas[type_noise_level].repeat_interleave(
                batch.num_atoms_t, dim=0))

        # add noise to atom types and sample atom types.
        pred_composition_probs = F.softmax(
            pred_composition_per_atom.detach(), dim=-1)
        atom_type_probs = (
                F.one_hot(batch.atom_types_t - 1, num_classes=MAX_ATOMIC_NUM) +
                pred_composition_probs * used_type_sigmas_per_atom[:, None])
        rand_atom_types = torch.multinomial(
            atom_type_probs, num_samples=1).squeeze(1) + 1

        # add noise to the cart coords
        cart_noises_per_atom = (
                torch.randn_like(batch.frac_coords_t) *
                used_sigmas_per_atom[:, None])
        cart_coords = frac_to_cart_coords(
            batch.frac_coords_t, pred_lengths, pred_angles, batch.num_atoms_t)
        cart_coords = cart_coords + cart_noises_per_atom
        noisy_frac_coords = cart_to_frac_coords(
            cart_coords, pred_lengths, pred_angles, batch.num_atoms_t)
        h, pred_cart_coord_diff = self.decoder(
            z, noisy_frac_coords, rand_atom_types, batch.num_atoms_t, pred_lengths, pred_angles)
        pred_atom_types = self.fc_atom(h)
        # compute loss.
        num_atom_loss = self.num_atom_loss(pred_num_atoms, batch)
        lattice_loss = self.lattice_loss(pred_lengths_and_angles, batch)
        composition_loss = self.composition_loss(
            pred_composition_per_atom, batch.atom_types_t, batch.atom_types_t_batch)
        coord_loss = self.coord_loss(
            pred_cart_coord_diff, noisy_frac_coords, used_sigmas_per_atom, batch)
        type_loss = self.type_loss(pred_atom_types, batch.atom_types_t,
                                   used_type_sigmas_per_atom )

        kld_loss = self.kld_loss(mu, log_var)

        if cfg.VAE.predict_property:
            property_loss = self.property_loss(z, batch)
        else:
            property_loss = 0.

        return {
            'num_atom_loss': num_atom_loss,
            'lattice_loss': lattice_loss,
            'composition_loss': composition_loss,
            'coord_loss': coord_loss,
            'type_loss': type_loss,
            'kld_loss': kld_loss,
            'property_loss': property_loss,
            'pred_num_atoms': pred_num_atoms,
            'pred_lengths_and_angles': pred_lengths_and_angles,
            'pred_lengths': pred_lengths,
            'pred_angles': pred_angles,
            'pred_cart_coord_diff': pred_cart_coord_diff,
            'pred_atom_types': pred_atom_types,
            'pred_composition_per_atom': pred_composition_per_atom,
            'target_frac_coords': batch.frac_coords_t,
            'target_atom_types': batch.atom_types_t,
            'rand_frac_coords': noisy_frac_coords,
            'rand_atom_types': rand_atom_types,
            'z': z,
        }
    def compute_stats(self, batch, outputs):

        num_atom_loss = outputs['num_atom_loss']
        lattice_loss = outputs['lattice_loss']
        coord_loss = outputs['coord_loss']
        type_loss = outputs['type_loss']
        kld_loss = outputs['kld_loss']
        composition_loss = outputs['composition_loss']
        property_loss = outputs['property_loss']


        loss = (
            cfg.VAE.cost_natom * num_atom_loss +
            cfg.VAE.cost_lattice * lattice_loss +
            cfg.VAE.cost_coord * coord_loss +
            cfg.VAE.cost_type * type_loss +
            cfg.VAE.cost_kld * kld_loss +
            cfg.VAE.cost_composition * composition_loss +
            cfg.VAE.cost_property * property_loss
        )

        return loss


    def num_atom_loss(self, pred_num_atoms, batch):
        return F.cross_entropy(pred_num_atoms, batch.num_atoms_t)

    def property_loss(self, z, batch):
        return F.mse_loss(self.fc_property(z), batch.y)

    def lattice_loss(self, pred_lengths_and_angles, batch):
        self.lattice_scaler.match_device(pred_lengths_and_angles)
        target_lengths = batch.lengths_t / \
                         batch.num_atoms_t.view(-1, 1).float() ** (1 / 3)
        target_lengths_and_angles = torch.cat(
            [target_lengths, batch.angles_t], dim=-1)
        target_lengths_and_angles = self.lattice_scaler.transform(
            target_lengths_and_angles)
        return F.mse_loss(pred_lengths_and_angles, target_lengths_and_angles)

    def composition_loss(self, pred_composition_per_atom, target_atom_types, batch):
        target_atom_types = target_atom_types - 1
        loss = F.cross_entropy(pred_composition_per_atom,
                               target_atom_types, reduction='none')
        return scatter(loss, batch, reduce='mean').mean()

    def coord_loss(self, pred_cart_coord_diff, noisy_frac_coords,
                   used_sigmas_per_atom, batch):
        noisy_cart_coords = frac_to_cart_coords(
            noisy_frac_coords, batch.lengths_t, batch.angles_t, batch.num_atoms_t)
        target_cart_coords = frac_to_cart_coords(
            batch.frac_coords_t, batch.lengths_t, batch.angles_t, batch.num_atoms_t)
        _, target_cart_coord_diff = min_distance_sqr_pbc(
            target_cart_coords, noisy_cart_coords, batch.lengths_t, batch.angles_t,
            batch.num_atoms_t, self.device, return_vector=True)

        target_cart_coord_diff = target_cart_coord_diff / \
                                 used_sigmas_per_atom[:, None] ** 2
        pred_cart_coord_diff = pred_cart_coord_diff / \
                               used_sigmas_per_atom[:, None]

        loss_per_atom = torch.sum(
            (target_cart_coord_diff - pred_cart_coord_diff) ** 2, dim=1)

        loss_per_atom = 0.5 * loss_per_atom * used_sigmas_per_atom ** 2
        return scatter(loss_per_atom, batch.atom_types_t_batch, reduce='mean').mean()

    def type_loss(self, pred_atom_types, target_atom_types,
                  used_type_sigmas_per_atom):
        target_atom_types = target_atom_types - 1
        loss = F.cross_entropy(
            pred_atom_types, target_atom_types, reduction='none')
        # rescale loss according to noise
        loss = loss / used_type_sigmas_per_atom
        return scatter(loss, batch.atom_types_t_batch, reduce='mean').mean()

    def kld_loss(self, mu, log_var):
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        return kld_loss


class latticeDEC(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.decoder = Decoder()
        self.fc_coord = build_mlp(args.hidden_features, args.hidden_features, 3, 3)
        self.fc_atom = nn.Linear(512,MAX_ATOMIC_NUM )
        self.device = args.device
    def predict_coord(self, n_z, num_atoms):
        z_per_atom = n_z.repeat_interleave(num_atoms, dim=0)
        pred_coords_per_atom = self.fc_coord(z_per_atom)
        return pred_coords_per_atom
    def forward(self, nz, decode_stats, batch):
        rep_type = torch.LongTensor().to(self.device)
        for i in range(len(batch)):
            b = batch[i]
            cnt = torch.unique_consecutive(b.target_atom_types, return_counts = True)[1]
            rep_type = torch.cat((rep_type, cnt))
        assert rep_type.sum() == batch.target_num_atoms.sum()
        pseudo_mean_coord = self.predict_coord(nz,rep_type)
        coord_var = torch.nn.Parameter(torch.Tensor(pseudo_mean_coord.shape)).to(self.device)
        pseudo_cart_coord = pseudo_mean_coord + coord_var
        (pred_num_atoms, pred_lengths_and_angles, pred_lengths, pred_angles, pred_composition_per_atom) = decode_stats
        # what atom type, 3D? or predicted atom type or 2D <- use 2D,

        h,  pred_cart_coord= self.decoder(
        nz, pseudo_cart_coord, batch.atom_types, batch.num_atoms, pred_lengths, pred_angles)
        pred_atom_types = self.fc_atom(h)
        return pred_cart_coord, pred_atom_types


if __name__ == "__main__":
    torch.manual_seed(cfg.RANDOM_SEED)
    from dataset import MaterialLoader
    train_loader, valid_loader, test_loader = MaterialLoader()
    model = crysVAE()
    model = model.to(model.device)
    from torch import optim
    import numpy as np
    optimizer = optim.Adam(params= model.parameters(),
                           lr = cfg.OPTIM.lr)

    for e in range(cfg.TRAIN.max_epoch):
        loss = []
        for batch in train_loader:
            batch.to(model.device)
            outputs = model(batch)
            running_loss = model.compute_stats(batch, outputs)
            running_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss.append(running_loss.item())
            print(f"running loss: {running_loss.item():.3f}")
        if (e+1) % 10 == 0:
            print(f"epoch {e+1}, loss: {np.mean(loss):.3f}")
            # n_z = n_z.detach()
            # decode_stat = (v.detach() for v in decode_stat)
            # pseudo_coord, pred_mean, pred_atom_type = lattDec(n_z, decode_stat,batch)
            # loss2 = coordLoss(pred_mean, batch)
            # loss2.backward()
            # optimizer_dec.step()
            # optimizer_dec.zero_grad()
            # print(f"coord loss: {loss2.item()}")
    # if (e+1) == 100:
    #     print(f"finished")
    #     model.eval()
    #     losses = []
    #     with torch.no_grad():
    #         for batch in test_loader:
    #             batch.to(args.device)
    #             nz, decode_stat, kld_loss = model(batch)
    #             loss1 = lattLoss(decode_stat, batch)
    #             loss1 += kld_loss
    #             losses.append(loss1.item())
    #         print(f"{np.mean(losses): .3f}")
    #         pred_coord, pred_type = lattDec(nz, decode_stat, batch)
