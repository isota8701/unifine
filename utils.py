from typing import Optional
import numpy as np
import torch
from torch import nn
import re
import copy

OFFSET_LIST = [
    [-1, -1, -1],
    [-1, -1, 0],
    [-1, -1, 1],
    [-1, 0, -1],
    [-1, 0, 0],
    [-1, 0, 1],
    [-1, 1, -1],
    [-1, 1, 0],
    [-1, 1, 1],
    [0, -1, -1],
    [0, -1, 0],
    [0, -1, 1],
    [0, 0, -1],
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, -1],
    [0, 1, 0],
    [0, 1, 1],
    [1, -1, -1],
    [1, -1, 0],
    [1, -1, 1],
    [1, 0, -1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, -1],
    [1, 1, 0],
    [1, 1, 1],
]
MAX_ATOMIC_NUM = 100

EPSILON = 1e-5

class RBFExpansion(nn.Module):
    """Expand interatomic distances with radial basis functions."""

    def __init__(
        self,
        vmin: float = 0,
        vmax: float = 8,
        bins: int = 40,
        lengthscale: Optional[float] = None,
    ):
        """Register torch parameters for RBF expansion."""
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.register_buffer(
            "centers", torch.linspace(self.vmin, self.vmax, self.bins)
        )

        if lengthscale is None:
            # SchNet-style
            # set lengthscales relative to granularity of RBF expansion
            self.lengthscale = np.diff(self.centers).mean()
            self.gamma = 1 / self.lengthscale

        else:
            self.lengthscale = lengthscale
            self.gamma = 1 / (lengthscale ** 2)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        """Apply RBF expansion to interatomic distance tensor."""
        return torch.exp(
            -self.gamma * (distance.unsqueeze(1) - self.centers) ** 2
        )

class Node(object):
    """ Node class for tree data structure """

    def __init__(self, parent, val=None):
        self.value = val
        self.parent = parent
        self.children = []

    def __repr__(self):
        return f"<Node {self.value} >"


def format_composition(comp):
    """ format str to ensure weights are explicate
    example: BaCu3 -> Ba1Cu3
    """
    subst = r"\g<1>1.0"
    comp = re.sub(r"[\d.]+", lambda x: str(float(x.group())), comp.rstrip())
    comp = re.sub(r"([A-Z][a-z](?![0-9]))", subst, comp)
    comp = re.sub(r"([A-Z](?![0-9]|[a-z]))", subst, comp)
    comp = re.sub(r"([\)](?=[A-Z]))", subst, comp)
    comp = re.sub(r"([\)](?=\())", subst, comp)
    return comp


def parenthetic_contents(string):
    """
    Generate parenthesized contents in string as (level, contents, weight).
    """
    num_after_bracket = r"[^0-9.]"

    stack = []
    for i, c in enumerate(string):
        if c == "(":
            stack.append(i)
        elif c == ")" and stack:
            start = stack.pop()
            num = re.split(num_after_bracket, string[i + 1:])[0] or 1
            yield {
                "value": [string[start + 1: i], float(num), False],
                "level": len(stack) + 1,
            }

    yield {"value": [string, 1, False], "level": 0}


def build_tree(root, data):
    """ build a tree from ordered levelled data """
    for record in data:
        last = root
        for _ in range(record["level"]):
            last = last.children[-1]
        last.children.append(Node(last, record["value"]))


def update_weights(comp, weight):
    """ split composition string into elements (keys) and weights
    example: Ba1Cu3 -> [Ba,Cu] [1,3]
    """
    regex3 = r"(\d+\.\d+)|(\d+)"
    parsed = [j for j in re.split(regex3, comp) if j]
    elements = parsed[0::2]
    weights = [float(p) * weight for p in parsed[1::2]]
    new_comp = ""
    for m, n in zip(elements, weights):
        new_comp += m + f"{n:.2f}"
    return new_comp


def update_parent(child):
    """ update the str for parent """
    input_str = child.value[2] or child.value[0]
    new_str = update_weights(input_str, child.value[1])
    pattern = re.escape("(" + child.value[0] + ")" + str(child.value[1]))
    old_str = child.parent.value[2] or child.parent.value[0]
    child.parent.value[2] = re.sub(pattern, new_str, old_str, 0)


def reduce_tree(current):
    """ perform a post-order reduction on the tree """
    if not current:
        pass

    for child in current.children:
        reduce_tree(child)
        update_parent(child)


def splitout_weights(comp):
    """ split composition string into elements (keys) and weights
    example: Ba1Cu3 -> [Ba,Cu] [1,3]
    """
    elements = []
    weights = []
    regex3 = r"(\d+\.\d+)|(\d+)"
    try:
        parsed = [j for j in re.split(regex3, comp) if j]
    except:
        print("parsed:", comp)
    elements += parsed[0::2]
    weights += parsed[1::2]
    weights = [float(w) for w in weights]
    return elements, weights


def parse_roost(string):
    # format the string to remove edge cases
    string = format_composition(string)
    # get nested bracket structure
    nested_levels = list(parenthetic_contents(string))
    if len(nested_levels) > 1:
        # reverse nested list
        nested_levels = nested_levels[::-1]
        # plant and grow the tree
        root = Node("root", ["None"] * 3)
        build_tree(root, nested_levels)
        # reduce the tree to get compositions
        reduce_tree(root)
        return splitout_weights(root.children[0].value[2])

    else:
        return splitout_weights(string)


def frac_to_cart_coords(
        frac_coords,
        lengths,
        angles,
        num_atoms,
):
    lattice = lattice_params_to_matrix_torch(lengths, angles)
    lattice_nodes = torch.repeat_interleave(lattice, num_atoms, dim=0)
    pos = torch.einsum('bi,bij->bj', frac_coords, lattice_nodes)  # cart coords

    return pos


def cart_to_frac_coords(
        cart_coords,
        lengths,
        angles,
        num_atoms,
):
    lattice = lattice_params_to_matrix_torch(lengths, angles)
    # use pinv in case the predicted lattice is not rank 3
    inv_lattice = torch.linalg.pinv(lattice)
    inv_lattice_nodes = torch.repeat_interleave(inv_lattice, num_atoms, dim=0)
    frac_coords = torch.einsum('bi,bij->bj', cart_coords, inv_lattice_nodes)
    return (frac_coords % 1.)


def get_pbc_distances(
        coords,
        edge_index,
        lengths,
        angles,
        to_jimages,
        num_atoms,
        num_bonds,
        coord_is_cart=False,
        return_offsets=False,
        return_distance_vec=False,
):
    lattice = lattice_params_to_matrix_torch(lengths, angles)

    if coord_is_cart:
        pos = coords
    else:
        lattice_nodes = torch.repeat_interleave(lattice, num_atoms, dim=0)
        pos = torch.einsum('bi,bij->bj', coords, lattice_nodes)  # cart coords

    j_index, i_index = edge_index

    distance_vectors = pos[j_index] - pos[i_index]

    # correct for pbc
    lattice_edges = torch.repeat_interleave(lattice, num_bonds, dim=0)
    offsets = torch.einsum('bi,bij->bj', to_jimages.float(), lattice_edges)
    distance_vectors += offsets

    # compute distances
    distances = distance_vectors.norm(dim=-1)

    out = {
        "edge_index": edge_index,
        "distances": distances,
    }

    if return_distance_vec:
        out["distance_vec"] = distance_vectors

    if return_offsets:
        out["offsets"] = offsets

    return out


def radius_graph_pbc_wrapper(data, radius, max_num_neighbors_threshold, device):
    cart_coords = frac_to_cart_coords(
        data.frac_coords, data.lengths, data.angles, data.num_atoms)
    return radius_graph_pbc(
        cart_coords, data.lengths, data.angles, data.num_atoms, radius,
        max_num_neighbors_threshold, device)


def radius_graph_pbc(cart_coords, lengths, angles, num_atoms,
                     radius, max_num_neighbors_threshold, device,
                     topk_per_pair=None):
    """Computes pbc graph edges under pbc.

    topk_per_pair: (num_atom_pairs,), select topk edges per atom pair

    Note: topk should take into account self-self edge for (i, i)
    """
    batch_size = len(num_atoms)

    # position of the atoms
    atom_pos = cart_coords

    # Before computing the pairwise distances between atoms, first create a list of atom indices to compare for the entire batch
    num_atoms_per_image = num_atoms
    num_atoms_per_image_sqr = (num_atoms_per_image ** 2).long()

    # index offset between images
    index_offset = (
            torch.cumsum(num_atoms_per_image, dim=0) - num_atoms_per_image
    )

    index_offset_expand = torch.repeat_interleave(
        index_offset, num_atoms_per_image_sqr
    )
    num_atoms_per_image_expand = torch.repeat_interleave(
        num_atoms_per_image, num_atoms_per_image_sqr
    )

    # Compute a tensor containing sequences of numbers that range from 0 to num_atoms_per_image_sqr for each image
    # that is used to compute indices for the pairs of atoms. This is a very convoluted way to implement
    # the following (but 10x faster since it removes the for loop)
    # for batch_idx in range(batch_size):
    #    batch_count = torch.cat([batch_count, torch.arange(num_atoms_per_image_sqr[batch_idx], device=device)], dim=0)
    num_atom_pairs = torch.sum(num_atoms_per_image_sqr)
    index_sqr_offset = (
            torch.cumsum(num_atoms_per_image_sqr, dim=0) - num_atoms_per_image_sqr
    )
    index_sqr_offset = torch.repeat_interleave(
        index_sqr_offset, num_atoms_per_image_sqr
    )
    atom_count_sqr = (
            torch.arange(num_atom_pairs, device=device) - index_sqr_offset
    )

    # Compute the indices for the pairs of atoms (using division and mod)
    # If the systems get too large this apporach could run into numerical precision issues
    index1 = (
                 (atom_count_sqr // num_atoms_per_image_expand)
             ).long() + index_offset_expand
    index2 = (
                     atom_count_sqr % num_atoms_per_image_expand
             ).long() + index_offset_expand
    # Get the positions for each atom
    pos1 = torch.index_select(atom_pos, 0, index1)
    pos2 = torch.index_select(atom_pos, 0, index2)

    unit_cell = torch.tensor(OFFSET_LIST, device=device).float()
    num_cells = len(unit_cell)
    unit_cell_per_atom = unit_cell.view(1, num_cells, 3).repeat(
        len(index2), 1, 1
    )
    unit_cell = torch.transpose(unit_cell, 0, 1)
    unit_cell_batch = unit_cell.view(1, 3, num_cells).expand(
        batch_size, -1, -1
    )

    # lattice matrix
    lattice = lattice_params_to_matrix_torch(lengths, angles)

    # Compute the x, y, z positional offsets for each cell in each image
    data_cell = torch.transpose(lattice, 1, 2)
    pbc_offsets = torch.bmm(data_cell, unit_cell_batch)
    pbc_offsets_per_atom = torch.repeat_interleave(
        pbc_offsets, num_atoms_per_image_sqr, dim=0
    )

    # Expand the positions and indices for the 9 cells
    pos1 = pos1.view(-1, 3, 1).expand(-1, -1, num_cells)
    pos2 = pos2.view(-1, 3, 1).expand(-1, -1, num_cells)
    index1 = index1.view(-1, 1).repeat(1, num_cells).view(-1)
    index2 = index2.view(-1, 1).repeat(1, num_cells).view(-1)
    # Add the PBC offsets for the second atom
    pos2 = pos2 + pbc_offsets_per_atom

    # Compute the squared distance between atoms
    atom_distance_sqr = torch.sum((pos1 - pos2) ** 2, dim=1)

    if topk_per_pair is not None:
        assert topk_per_pair.size(0) == num_atom_pairs
        atom_distance_sqr_sort_index = torch.argsort(atom_distance_sqr, dim=1)
        assert atom_distance_sqr_sort_index.size() == (num_atom_pairs, num_cells)
        atom_distance_sqr_sort_index = (
                atom_distance_sqr_sort_index +
                torch.arange(num_atom_pairs, device=device)[:, None] * num_cells).view(-1)
        topk_mask = (torch.arange(num_cells, device=device)[None, :] <
                     topk_per_pair[:, None])
        topk_mask = topk_mask.view(-1)
        topk_indices = atom_distance_sqr_sort_index.masked_select(topk_mask)

        topk_mask = torch.zeros(num_atom_pairs * num_cells, device=device)
        topk_mask.scatter_(0, topk_indices, 1.)
        topk_mask = topk_mask.bool()

    atom_distance_sqr = atom_distance_sqr.view(-1)

    # Remove pairs that are too far apart
    mask_within_radius = torch.le(atom_distance_sqr, radius * radius)
    # Remove pairs with the same atoms (distance = 0.0)
    mask_not_same = torch.gt(atom_distance_sqr, 0.0001)
    mask = torch.logical_and(mask_within_radius, mask_not_same)
    index1 = torch.masked_select(index1, mask)
    index2 = torch.masked_select(index2, mask)
    unit_cell = torch.masked_select(
        unit_cell_per_atom.view(-1, 3), mask.view(-1, 1).expand(-1, 3)
    )
    unit_cell = unit_cell.view(-1, 3)
    if topk_per_pair is not None:
        topk_mask = torch.masked_select(topk_mask, mask)

    num_neighbors = torch.zeros(len(cart_coords), device=device)
    num_neighbors.index_add_(0, index1, torch.ones(len(index1), device=device))
    num_neighbors = num_neighbors.long()
    max_num_neighbors = torch.max(num_neighbors).long()

    # Compute neighbors per image
    _max_neighbors = copy.deepcopy(num_neighbors)
    _max_neighbors[
        _max_neighbors > max_num_neighbors_threshold
        ] = max_num_neighbors_threshold
    _num_neighbors = torch.zeros(len(cart_coords) + 1, device=device).long()
    _natoms = torch.zeros(num_atoms.shape[0] + 1, device=device).long()
    _num_neighbors[1:] = torch.cumsum(_max_neighbors, dim=0)
    _natoms[1:] = torch.cumsum(num_atoms, dim=0)
    num_neighbors_image = (
            _num_neighbors[_natoms[1:]] - _num_neighbors[_natoms[:-1]]
    )

    # If max_num_neighbors is below the threshold, return early
    if (
            max_num_neighbors <= max_num_neighbors_threshold
            or max_num_neighbors_threshold <= 0
    ):
        if topk_per_pair is None:
            return torch.stack((index2, index1)), unit_cell, num_neighbors_image
        else:
            return torch.stack((index2, index1)), unit_cell, num_neighbors_image, topk_mask

    atom_distance_sqr = torch.masked_select(atom_distance_sqr, mask)

    # Create a tensor of size [num_atoms, max_num_neighbors] to sort the distances of the neighbors.
    # Fill with values greater than radius*radius so we can easily remove unused distances later.
    distance_sort = torch.zeros(
        len(cart_coords) * max_num_neighbors, device=device
    ).fill_(radius * radius + 1.0)

    # Create an index map to map distances from atom_distance_sqr to distance_sort
    index_neighbor_offset = torch.cumsum(num_neighbors, dim=0) - num_neighbors
    index_neighbor_offset_expand = torch.repeat_interleave(
        index_neighbor_offset, num_neighbors
    )
    index_sort_map = (
            index1 * max_num_neighbors
            + torch.arange(len(index1), device=device)
            - index_neighbor_offset_expand
    )
    distance_sort.index_copy_(0, index_sort_map, atom_distance_sqr)
    distance_sort = distance_sort.view(len(cart_coords), max_num_neighbors)

    # Sort neighboring atoms based on distance
    distance_sort, index_sort = torch.sort(distance_sort, dim=1)
    # Select the max_num_neighbors_threshold neighbors that are closest
    distance_sort = distance_sort[:, :max_num_neighbors_threshold]
    index_sort = index_sort[:, :max_num_neighbors_threshold]

    # Offset index_sort so that it indexes into index1
    index_sort = index_sort + index_neighbor_offset.view(-1, 1).expand(
        -1, max_num_neighbors_threshold
    )
    # Remove "unused pairs" with distances greater than the radius
    mask_within_radius = torch.le(distance_sort, radius * radius)
    index_sort = torch.masked_select(index_sort, mask_within_radius)

    # At this point index_sort contains the index into index1 of the closest max_num_neighbors_threshold neighbors per atom
    # Create a mask to remove all pairs not in index_sort
    mask_num_neighbors = torch.zeros(len(index1), device=device).bool()
    mask_num_neighbors.index_fill_(0, index_sort, True)

    # Finally mask out the atoms to ensure each atom has at most max_num_neighbors_threshold neighbors
    index1 = torch.masked_select(index1, mask_num_neighbors)
    index2 = torch.masked_select(index2, mask_num_neighbors)
    unit_cell = torch.masked_select(
        unit_cell.view(-1, 3), mask_num_neighbors.view(-1, 1).expand(-1, 3)
    )
    unit_cell = unit_cell.view(-1, 3)

    if topk_per_pair is not None:
        topk_mask = torch.masked_select(topk_mask, mask_num_neighbors)

    edge_index = torch.stack((index2, index1))

    if topk_per_pair is None:
        return edge_index, unit_cell, num_neighbors_image
    else:
        return edge_index, unit_cell, num_neighbors_image, topk_mask


def min_distance_sqr_pbc(cart_coords1, cart_coords2, lengths, angles,
                         num_atoms, device, return_vector=False,
                         return_to_jimages=False):
    """Compute the pbc distance between atoms in cart_coords1 and cart_coords2.
    This function assumes that cart_coords1 and cart_coords2 have the same number of atoms
    in each data point.
    returns:
        basic return:
            min_atom_distance_sqr: (N_atoms, )
        return_vector == True:
            min_atom_distance_vector: vector pointing from cart_coords1 to cart_coords2, (N_atoms, 3)
        return_to_jimages == True:
            to_jimages: (N_atoms, 3), position of cart_coord2 relative to cart_coord1 in pbc
    """
    batch_size = len(num_atoms)

    # Get the positions for each atom
    pos1 = cart_coords1
    pos2 = cart_coords2

    unit_cell = torch.tensor(OFFSET_LIST, device=device).float()
    num_cells = len(unit_cell)
    unit_cell_per_atom = unit_cell.view(1, num_cells, 3).repeat(
        len(cart_coords2), 1, 1
    )
    unit_cell = torch.transpose(unit_cell, 0, 1)
    unit_cell_batch = unit_cell.view(1, 3, num_cells).expand(
        batch_size, -1, -1
    )

    # lattice matrix
    lattice = lattice_params_to_matrix_torch(lengths, angles)

    # Compute the x, y, z positional offsets for each cell in each image
    data_cell = torch.transpose(lattice, 1, 2)
    pbc_offsets = torch.bmm(data_cell, unit_cell_batch)
    pbc_offsets_per_atom = torch.repeat_interleave(
        pbc_offsets, num_atoms, dim=0
    )

    # Expand the positions and indices for the 9 cells
    pos1 = pos1.view(-1, 3, 1).expand(-1, -1, num_cells)
    pos2 = pos2.view(-1, 3, 1).expand(-1, -1, num_cells)
    # Add the PBC offsets for the second atom
    pos2 = pos2 + pbc_offsets_per_atom

    # Compute the vector between atoms
    # shape (num_atom_squared_sum, 3, 27)
    atom_distance_vector = pos1 - pos2
    atom_distance_sqr = torch.sum(atom_distance_vector ** 2, dim=1)

    min_atom_distance_sqr, min_indices = atom_distance_sqr.min(dim=-1)

    return_list = [min_atom_distance_sqr]

    if return_vector:
        min_indices = min_indices[:, None, None].repeat([1, 3, 1])

        min_atom_distance_vector = torch.gather(
            atom_distance_vector, 2, min_indices).squeeze(-1)

        return_list.append(min_atom_distance_vector)

    if return_to_jimages:
        to_jimages = unit_cell.T[min_indices].long()
        return_list.append(to_jimages)

    return return_list[0] if len(return_list) == 1 else return_list

def lattice_params_to_matrix_torch(lengths, angles):
    """Batched torch version to compute lattice matrix from params.
    lengths: torch.Tensor of shape (N, 3), unit A
    angles: torch.Tensor of shape (N, 3), unit degree
    """
    angles_r = torch.deg2rad(angles)
    coses = torch.cos(angles_r)
    sins = torch.sin(angles_r)

    val = (coses[:, 0] * coses[:, 1] - coses[:, 2]) / (sins[:, 0] * sins[:, 1])
    # Sometimes rounding errors result in values slightly > 1.
    val = torch.clamp(val, -1., 1.)
    gamma_star = torch.arccos(val)

    vector_a = torch.stack([
        lengths[:, 0] * sins[:, 1],
        torch.zeros(lengths.size(0), device=lengths.device),
        lengths[:, 0] * coses[:, 1]], dim=1)
    vector_b = torch.stack([
        -lengths[:, 1] * sins[:, 0] * torch.cos(gamma_star),
        lengths[:, 1] * sins[:, 0] * torch.sin(gamma_star),
        lengths[:, 1] * coses[:, 0]], dim=1)
    vector_c = torch.stack([
        torch.zeros(lengths.size(0), device=lengths.device),
        torch.zeros(lengths.size(0), device=lengths.device),
        lengths[:, 2]], dim=1)

    return torch.stack([vector_a, vector_b, vector_c], dim=1)



class StandardScalerTorch(object):
    """Normalizes the targets of a dataset."""

    def __init__(self, means=None, stds=None):
        self.means = means
        self.stds = stds

    def fit(self, X):
        # X = torch.tensor(X, dtype=torch.float)
        self.means = torch.mean(X, dim=0)
        # https://github.com/pytorch/pytorch/issues/29372
        self.stds = torch.std(X, dim=0, unbiased=False) + EPSILON

    def transform(self, X):
        # X = torch.tensor(X, dtype=torch.float)
        return (X - self.means) / self.stds

    def inverse_transform(self, X):
        # X = torch.tensor(X, dtype=torch.float)
        return X * self.stds + self.means

    def match_device(self, tensor):
        if self.means.device != tensor.device:
            self.means = self.means.to(tensor.device)
            self.stds = self.stds.to(tensor.device)

    def copy(self):
        return StandardScalerTorch(
            means=self.means.clone().detach(),
            stds=self.stds.clone().detach())

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"means: {self.means.tolist()}, "
            f"stds: {self.stds.tolist()})"
        )



def get_scaler_from_data_list(data_list, key):
    targets = torch.tensor(np.array([d[key] for d in data_list]), dtype = torch.float)
    scaler = StandardScalerTorch()
    scaler.fit(targets)
    return scaler


"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import json

import torch
from torch_scatter import segment_csr


def read_json(path):
    """"""
    if not path.endswith(".json"):
        raise UserWarning(f"Path {path} is not a json-path.")

    with open(path, "r") as f:
        content = json.load(f)
    return content


def update_json(path, data):
    """"""
    if not path.endswith(".json"):
        raise UserWarning(f"Path {path} is not a json-path.")

    content = read_json(path)
    content.update(data)
    write_json(path, content)


def write_json(path, data):
    """"""
    if not path.endswith(".json"):
        raise UserWarning(f"Path {path} is not a json-path.")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def read_value_json(path, key):
    """"""
    content = read_json(path)

    if key in content.keys():
        return content[key]
    else:
        return None


def ragged_range(sizes):
    """Multiple concatenated ranges.

    Examples
    --------
        sizes = [1 4 2 3]
        Return: [0  0 1 2 3  0 1  0 1 2]
    """
    assert sizes.dim() == 1
    if sizes.sum() == 0:
        return sizes.new_empty(0)

    # Remove 0 sizes
    sizes_nonzero = sizes > 0
    if not torch.all(sizes_nonzero):
        sizes = torch.masked_select(sizes, sizes_nonzero)

    # Initialize indexing array with ones as we need to setup incremental indexing
    # within each group when cumulatively summed at the final stage.
    id_steps = torch.ones(sizes.sum(), dtype=torch.long, device=sizes.device)
    id_steps[0] = 0
    insert_index = sizes[:-1].cumsum(0)
    insert_val = (1 - sizes)[:-1]

    # Assign index-offsetting values
    id_steps[insert_index] = insert_val

    # Finally index into input array for the group repeated o/p
    res = id_steps.cumsum(0)
    return res


def repeat_blocks(
        sizes,
        repeats,
        continuous_indexing=True,
        start_idx=0,
        block_inc=0,
        repeat_inc=0,
):
    """Repeat blocks of indices.
    Adapted from https://stackoverflow.com/questions/51154989/numpy-vectorized-function-to-repeat-blocks-of-consecutive-elements

    continuous_indexing: Whether to keep increasing the index after each block
    start_idx: Starting index
    block_inc: Number to increment by after each block,
               either global or per block. Shape: len(sizes) - 1
    repeat_inc: Number to increment by after each repetition,
                either global or per block

    Examples
    --------
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = False
        Return: [0 0 0  0 1 2 0 1 2  0 1 0 1 0 1]
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True
        Return: [0 0 0  1 2 3 1 2 3  4 5 4 5 4 5]
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True ;
        repeat_inc = 4
        Return: [0 4 8  1 2 3 5 6 7  4 5 8 9 12 13]
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True ;
        start_idx = 5
        Return: [5 5 5  6 7 8 6 7 8  9 10 9 10 9 10]
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True ;
        block_inc = 1
        Return: [0 0 0  2 3 4 2 3 4  6 7 6 7 6 7]
        sizes = [0,3,2] ; repeats = [3,2,3] ; continuous_indexing = True
        Return: [0 1 2 0 1 2  3 4 3 4 3 4]
        sizes = [2,3,2] ; repeats = [2,0,2] ; continuous_indexing = True
        Return: [0 1 0 1  5 6 5 6]
    """
    assert sizes.dim() == 1
    assert all(sizes >= 0)

    # Remove 0 sizes
    sizes_nonzero = sizes > 0
    if not torch.all(sizes_nonzero):
        assert block_inc == 0  # Implementing this is not worth the effort
        sizes = torch.masked_select(sizes, sizes_nonzero)
        if isinstance(repeats, torch.Tensor):
            repeats = torch.masked_select(repeats, sizes_nonzero)
        if isinstance(repeat_inc, torch.Tensor):
            repeat_inc = torch.masked_select(repeat_inc, sizes_nonzero)

    if isinstance(repeats, torch.Tensor):
        assert all(repeats >= 0)
        insert_dummy = repeats[0] == 0
        if insert_dummy:
            one = sizes.new_ones(1)
            zero = sizes.new_zeros(1)
            sizes = torch.cat((one, sizes))
            repeats = torch.cat((one, repeats))
            if isinstance(block_inc, torch.Tensor):
                block_inc = torch.cat((zero, block_inc))
            if isinstance(repeat_inc, torch.Tensor):
                repeat_inc = torch.cat((zero, repeat_inc))
    else:
        assert repeats >= 0
        insert_dummy = False

    # Get repeats for each group using group lengths/sizes
    r1 = torch.repeat_interleave(
        torch.arange(len(sizes), device=sizes.device), repeats
    )

    # Get total size of output array, as needed to initialize output indexing array
    N = (sizes * repeats).sum()

    # Initialize indexing array with ones as we need to setup incremental indexing
    # within each group when cumulatively summed at the final stage.
    # Two steps here:
    # 1. Within each group, we have multiple sequences, so setup the offsetting
    # at each sequence lengths by the seq. lengths preceding those.
    id_ar = torch.ones(N, dtype=torch.long, device=sizes.device)
    id_ar[0] = 0
    insert_index = sizes[r1[:-1]].cumsum(0)
    insert_val = (1 - sizes)[r1[:-1]]

    if isinstance(repeats, torch.Tensor) and torch.any(repeats == 0):
        diffs = r1[1:] - r1[:-1]
        indptr = torch.cat((sizes.new_zeros(1), diffs.cumsum(0)))
        if continuous_indexing:
            # If a group was skipped (repeats=0) we need to add its size
            insert_val += segment_csr(sizes[: r1[-1]], indptr, reduce="sum")

        # Add block increments
        if isinstance(block_inc, torch.Tensor):
            insert_val += segment_csr(
                block_inc[: r1[-1]], indptr, reduce="sum"
            )
        else:
            insert_val += block_inc * (indptr[1:] - indptr[:-1])
            if insert_dummy:
                insert_val[0] -= block_inc
    else:
        idx = r1[1:] != r1[:-1]
        if continuous_indexing:
            # 2. For each group, make sure the indexing starts from the next group's
            # first element. So, simply assign 1s there.
            insert_val[idx] = 1

        # Add block increments
        insert_val[idx] += block_inc

    # Add repeat_inc within each group
    if isinstance(repeat_inc, torch.Tensor):
        insert_val += repeat_inc[r1[:-1]]
        if isinstance(repeats, torch.Tensor):
            repeat_inc_inner = repeat_inc[repeats > 0][:-1]
        else:
            repeat_inc_inner = repeat_inc[:-1]
    else:
        insert_val += repeat_inc
        repeat_inc_inner = repeat_inc

    # Subtract the increments between groups
    if isinstance(repeats, torch.Tensor):
        repeats_inner = repeats[repeats > 0][:-1]
    else:
        repeats_inner = repeats
    insert_val[r1[1:] != r1[:-1]] -= repeat_inc_inner * repeats_inner

    # Assign index-offsetting values
    id_ar[insert_index] = insert_val

    if insert_dummy:
        id_ar = id_ar[1:]
        if continuous_indexing:
            id_ar[0] -= 1

    # Set start index now, in case of insertion due to leading repeats=0
    id_ar[0] += start_idx

    # Finally index into input array for the group repeated o/p
    res = id_ar.cumsum(0)
    return res


def calculate_interatomic_vectors(R, id_s, id_t, offsets_st):
    """
    Calculate the vectors connecting the given atom pairs,
    considering offsets from periodic boundary conditions (PBC).

    Parameters
    ----------
        R: Tensor, shape = (nAtoms, 3)
            Atom positions.
        id_s: Tensor, shape = (nEdges,)
            Indices of the source atom of the edges.
        id_t: Tensor, shape = (nEdges,)
            Indices of the target atom of the edges.
        offsets_st: Tensor, shape = (nEdges,)
            PBC offsets of the edges.
            Subtract this from the correct direction.

    Returns
    -------
        (D_st, V_st): tuple
            D_st: Tensor, shape = (nEdges,)
                Distance from atom t to s.
            V_st: Tensor, shape = (nEdges,)
                Unit direction from atom t to s.
    """
    Rs = R[id_s]
    Rt = R[id_t]
    # ReLU prevents negative numbers in sqrt
    if offsets_st is None:
        V_st = Rt - Rs  # s -> t
    else:
        V_st = Rt - Rs + offsets_st  # s -> t
    D_st = torch.sqrt(torch.sum(V_st ** 2, dim=1))
    V_st = V_st / D_st[..., None]
    return D_st, V_st


def inner_product_normalized(x, y):
    """
    Calculate the inner product between the given normalized vectors,
    giving a result between -1 and 1.
    """
    return torch.sum(x * y, dim=-1).clamp(min=-1, max=1)


def mask_neighbors(neighbors, edge_mask):
    neighbors_old_indptr = torch.cat([neighbors.new_zeros(1), neighbors])
    neighbors_old_indptr = torch.cumsum(neighbors_old_indptr, dim=0)
    neighbors = segment_csr(edge_mask.long(), neighbors_old_indptr)
    return neighbors

