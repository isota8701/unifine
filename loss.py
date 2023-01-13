import torch
import torch.nn as nn
from torch_scatter import scatter
import torch.nn.functional as F

class LatticeLoss(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.alpha = args.alpha
        self.beta = args.beta
        self.gamma = args.gamma
        self.lattice_scaler = torch.load("./data/"+ "lattice_scaler.pt")

    def forward(self, decode_stats, batch):
        (pred_num_atoms, pred_lengths_and_angles, pred_lengths, pred_angles, pred_composition_per_atom) = decode_stats

        num_atom_loss = self.num_atom_loss(pred_num_atoms, batch)
        lattice_loss = self.lattice_loss(pred_lengths_and_angles, batch)
        composition_loss = self.composition_loss(
            pred_composition_per_atom, batch.atom_types, batch)
        loss = self.alpha*num_atom_loss + self.beta*lattice_loss + self.gamma* composition_loss
        return loss

    def num_atom_loss(self, pred_num_atoms, batch):
        return F.cross_entropy(pred_num_atoms, batch.num_atoms)

    def property_loss(self, z, batch):
        return F.mse_loss(self.fc_property(z), batch.y)

    def lattice_loss(self, pred_lengths_and_angles, batch):
        self.lattice_scaler.match_device(pred_lengths_and_angles)
        target_lengths = batch.lengths / \
                         batch.num_atoms.view(-1, 1).float() ** (1 / 3)
        target_lengths_and_angles = torch.cat(
            [target_lengths, batch.angles], dim=-1)
        target_lengths_and_angles = self.lattice_scaler.transform(
            target_lengths_and_angles)
        return F.mse_loss(pred_lengths_and_angles, target_lengths_and_angles)

    def composition_loss(self, pred_composition_per_atom, target_atom_types, batch):
        # target_atom_types = target_atom_types - 1
        loss = F.cross_entropy(pred_composition_per_atom,
                               target_atom_types, reduction='none')
        return scatter(loss, batch.batch, reduce='mean').mean()


# class CoordLoss(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#     def forward(self, pred_cart_coord, batch):
#
#
#
#
#     def coord_loss(self, pred_cart_coord_diff, noisy_frac_coords,
#                    used_sigmas_per_atom, batch):
#
#         target_cart_coords = frac_to_cart_coords(
#             batch.frac_coords, batch.lengths, batch.angles, batch.num_atoms)
#         _, target_cart_coord_diff = min_distance_sqr_pbc(
#             target_cart_coords, noisy_cart_coords, batch.lengths, batch.angles,
#             batch.num_atoms, self.device, return_vector=True)
#
#         target_cart_coord_diff = target_cart_coord_diff / \
#                                  used_sigmas_per_atom[:, None] ** 2
#         pred_cart_coord_diff = pred_cart_coord_diff / \
#                                used_sigmas_per_atom[:, None]
#
#         loss_per_atom = torch.sum(
#             (target_cart_coord_diff - pred_cart_coord_diff) ** 2, dim=1)
#
#         loss_per_atom = 0.5 * loss_per_atom * used_sigmas_per_atom ** 2
#         return scatter(loss_per_atom, batch.batch, reduce='mean').mean()
#
#     def type_loss(self, pred_atom_types, target_atom_types,
#                   used_type_sigmas_per_atom, batch):
#         target_atom_types = target_atom_types - 1
#         loss = F.cross_entropy(
#             pred_atom_types, target_atom_types, reduction='none')
#         # rescale loss according to noise
#         loss = loss / used_type_sigmas_per_atom
#         return scatter(loss, batch.batch, reduce='mean').mean()
#
