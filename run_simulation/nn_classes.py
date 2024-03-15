import torch
import sys
from schnetpack import Properties
import torch.nn as nn
import schnetpack as sch
from typing import Optional
from schnetpack.nn.acsf import gaussian_smearing
from schnetpack.nn.activations import shifted_softplus
from schnetpack.nn.cfconv import CFConv
from torch.nn import functional
from torch.nn.init import xavier_uniform_
from schnetpack.nn.initializers import zeros_initializer
from typing import Dict
from openmm.unit import *
from openmm.app import *
from openmm import *

class NNAtoms:
    """
    Class which obtains information about the residues which the neural network
    is applied to from the OpenMM system and simulation object
    """
    def __init__(self, simmd, system, nn_resid):
        """
        simmd : Simulation object
            OpenMM simulation object
        system : System object
            OpenMM system object
        nn_resid : list
            List of residue indices for which the neural network is applied to
        """
        self.simmd = simmd
        self.system = system
        self.nn_resid = nn_resid

        self.allNNAtomIndex = []
        self.allNNAtomElements = []
        self.realNNAtomIndex = []
        self.realNNAtomElements = []
        for res_index, residue in enumerate(simmd.topology.residues()):
            if res_index in nn_resid:
                for atom_index, atom in enumerate(residue.atoms()):
                    self.allNNAtomIndex.append(atom.index)
                    self.allNNAtomElements.append(atom.element)
                    if atom.element:
                        self.realNNAtomIndex.append(atom.index)
                        self.realNNAtomElements.append(atom.element)

        self.atomicNumbers = [i.atomic_number for i in self.realNNAtomElements]
        self.numRealAtoms = len(self.atomicNumbers)

    def generateNNExclusions(self):
        """
        Generate exclusions for all atoms for which the neural network is applied to
        """
        harmonicBondForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == HarmonicBondForce][0]
        harmonicAngleForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == HarmonicAngleForce][0]
        periodicTorsionForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == PeriodicTorsionForce][0]
        rbTorsionForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == RBTorsionForce][0]
        drudeForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == DrudeForce][0]
        nbondedForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == NonbondedForce][0]
        customNonbondedForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == CustomNonbondedForce][0]
        customBondForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == CustomBondForce][0]

        #Zero energies from intramolecular forces for residues with neural networks
        for i in range(harmonicBondForce.getNumBonds()):
            p1, p2, r0, k = harmonicBondForce.getBondParameters(i)
            if p1 in self.allNNAtomIndex or p2 in self.allNNAtomIndex:
                k = Quantity(0, unit=k.unit)
                harmonicBondForce.setBondParameters(i, p1, p2, r0, k)

        for i in range(harmonicAngleForce.getNumAngles()):
            p1, p2, p3, r0, k = harmonicAngleForce.getAngleParameters(i)
            if p1 in self.allNNAtomIndex or p2 in self.allNNAtomIndex or p3 in self.allNNAtomIndex:
                k = Quantity(0, unit=k.unit)
                harmonicAngleForce.setAngleParameters(i, p1, p2, p3, r0, k)

        for i in range(periodicTorsionForce.getNumTorsions()):
            p1, p2, p3, p4, period, r0, k = periodicTorsionForce.getTorsionParameters(i)
            if p1 in self.allNNAtomIndex or p2 in self.allNNAtomIndex or p3 in self.allNNAtomIndex or p4 in self.allNNAtomIndex:
                k = Quantity(0, unit=k.unit)
                periodicTorsionForce.setTorsionParameters(i, p1, p2, p3, p4, period, r0, k)

        for i in range(rbTorsionForce.getNumTorsions()):
            p1, p2, p3, p4, c1, c2, c3, c4, c5, c6 = rbTorsionForce.getTorsionParameters(i)
            if p1 in self.allNNAtomIndex or p2 in self.allNNAtomIndex or p3 in self.allNNAtomIndex or p4 in self.allNNAtomIndex:
                c1 = Quantity(0, unit=c1.unit)
                c2 = Quantity(0, unit=c2.unit)
                c3 = Quantity(0, unit=c3.unit)
                c4 = Quantity(0, unit=c4.unit)
                c5 = Quantity(0, unit=c5.unit)
                c6 = Quantity(0, unit=c6.unit)
                rbTorsionForce.setTorsionParameters(i, p1, p2, p3, p4, c1, c2, c3, c4, c5, c6)

        for i in range(customBondForce.getNumBonds()):
            p1, p2, parms = customBondForce.getBondParameters(i)
            if p1 not in self.allNNAtomIndex or p2 not in self.allNNAtomIndex:
                customBondForce.setBondParameters(i, p1, p2, (0.0, 0.1, 10.0))
            p1, p2, parms = customBondForce.getBondParameters(i)

        # map from global particle index to drudeforce object index
        particleMap = {}
        for i in range(drudeForce.getNumParticles()):
            particleMap[drudeForce.getParticleParameters(i)[0]] = i

        # can't add duplicate ScreenedPairs, so store what we already have
        flagexceptions = {}
        for i in range(nbondedForce.getNumExceptions()):
            (particle1, particle2, charge, sigma, epsilon) = nbondedForce.getExceptionParameters(i)
            string1=str(particle1)+"_"+str(particle2)
            string2=str(particle2)+"_"+str(particle1)
            flagexceptions[string1]=1
            flagexceptions[string2]=1

        # can't add duplicate customNonbonded exclusions, so store what we already have
        flagexclusions = {}
        for i in range(customNonbondedForce.getNumExclusions()):
            (particle1, particle2) = customNonbondedForce.getExclusionParticles(i)
            string1=str(particle1)+"_"+str(particle2)
            string2=str(particle2)+"_"+str(particle1)
            flagexclusions[string1]=1
            flagexclusions[string2]=1

        print(' adding exclusions ...')

        # add all intra-molecular exclusions, and when a drude pair is
        # excluded add a corresponding screened thole interaction in its place
        current_res = 0
        for res in self.simmd.topology.residues():
            if current_res in self.allNNAtomIndex:
                for i in range(len(res._atoms)-1):
                    for j in range(i+1,len(res._atoms)):
                        (indi,indj) = (res._atoms[i].index, res._atoms[j].index)
                        # here it doesn't matter if we already have this, since we pass the "True" flag
                        nbondedForce.addException(indi,indj,0,1,0,True)
                        # make sure we don't already exclude this customnonbond
                        string1=str(indi)+"_"+str(indj)
                        string2=str(indj)+"_"+str(indi)
                        if string1 in flagexclusions or string2 in flagexclusions:
                            continue
                        else:
                            customNonbondedForce.addExclusion(indi,indj)
                        # add thole if we're excluding two drudes
                        if indi in particleMap and indj in particleMap:
                            # make sure we don't already have this screened pair
                            if string1 in flagexceptions or string2 in flagexceptions:
                                continue
                            else:
                                drudei = particleMap[indi]
                                drudej = particleMap[indj]
                                drudeForce.addScreenedPair(drudei, drudej, 2.0)
            current_res += 1

        # now reinitialize to make sure changes are stored in context
        state = self.simmd.context.getState(getEnergy=False,getForces=False,getVelocities=False,getPositions=True)
        positions = state.getPositions()
        self.simmd.context.reinitialize()
        self.simmd.context.setPositions(positions)

@torch.jit.script
def neighbor_pairs(padding_mask, coordinates, cell, shifts, cutoff):
    """Compute pairs of atoms that are neighbors
    Copyright 2018- Xiang Gao and other ANI developers
    (https://github.com/aiqm/torchani/blob/master/torchani/aev.py)

    Arguments:
        padding_mask (:class:`torch.Tensor`): boolean tensor of shape
            (molecules, atoms) for padding mask. 1 == is padding.
        coordinates (:class:`torch.Tensor`): tensor of shape
            (molecules, atoms, 3) for atom coordinates.
        cell (:class:`torch.Tensor`): tensor of shape (3, 3) of the three vectors
            defining unit cell: tensor([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])
        cutoff (float): the cutoff inside which atoms are considered pairs
        shifts (:class:`torch.Tensor`): tensor of shape (?, 3) storing shifts
    """
    coordinates = coordinates.detach()
    cell = cell.detach()
    num_atoms = padding_mask.shape[0]
    all_atoms = torch.arange(num_atoms, device=cell.device)

    # Step 2: center cell
    p1_center, p2_center = torch.combinations(all_atoms).unbind(-1)
    shifts_center = shifts.new_zeros(p1_center.shape[0], 3)

    # Step 3: cells with shifts
    # shape convention (shift index, molecule index, atom index, 3)
    num_shifts = shifts.shape[0]
    all_shifts = torch.arange(num_shifts, device=cell.device)
    shift_index, p1, p2 = torch.cartesian_prod(all_shifts, all_atoms, all_atoms).unbind(
        -1
    )
    shifts_outside = shifts.index_select(0, shift_index)

    # Step 4: combine results for all cells
    shifts_all = torch.cat([shifts_center, shifts_outside])
    p1_all = torch.cat([p1_center, p1])
    p2_all = torch.cat([p2_center, p2])

    shift_values = torch.mm(shifts_all.to(cell.dtype), cell)

    # step 5, compute distances, and find all pairs within cutoff
    distances = (coordinates[p1_all] - coordinates[p2_all] + shift_values).norm(2, -1)

    padding_mask = (padding_mask[p1_all]) | (padding_mask[p2_all])
    distances.masked_fill_(padding_mask, torch.inf)
    in_cutoff = torch.nonzero(distances < cutoff)
    pair_index = in_cutoff.squeeze()
    atom_index1 = p1_all[pair_index]
    atom_index2 = p2_all[pair_index]
    shifts = shifts_all.index_select(0, pair_index)
    return atom_index1, atom_index2, shifts

@torch.jit.script
def compute_shifts(cell, pbc, cutoff):
    """Compute the shifts of unit cell along the given cell vectors to make it
    large enough to contain all pairs of neighbor atoms with PBC under
    consideration.
    Copyright 2018- Xiang Gao and other ANI developers
    (https://github.com/aiqm/torchani/blob/master/torchani/aev.py)

    Arguments:
        cell (:class:`torch.Tensor`): tensor of shape (3, 3) of the three
        vectors defining unit cell:
            tensor([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])
        cutoff (float): the cutoff inside which atoms are considered pairs
        pbc (:class:`torch.Tensor`): boolean vector of size 3 storing
            if pbc is enabled for that direction.

    Returns:
        :class:`torch.Tensor`: long tensor of shifts. the center cell and
            symmetric cells are not included.
    """
    
    reciprocal_cell = cell.inverse().t()
    inv_distances = reciprocal_cell.norm(2, -1)
    num_repeats = torch.ceil(cutoff * inv_distances).to(torch.long)
    num_repeats = torch.where(pbc, num_repeats, torch.zeros_like(num_repeats))

    r1 = torch.arange(1, num_repeats[0] + 1, device=cell.device)
    r2 = torch.arange(1, num_repeats[1] + 1, device=cell.device)
    r3 = torch.arange(1, num_repeats[2] + 1, device=cell.device)
    o = torch.zeros(1, dtype=torch.long, device=cell.device)
   
    return torch.cat(
        [
            torch.cartesian_prod(r1, r2, r3),
            torch.cartesian_prod(r1, r2, o),
            torch.cartesian_prod(r1, r2, -r3),
            torch.cartesian_prod(r1, o, r3),
            torch.cartesian_prod(r1, o, o),
            torch.cartesian_prod(r1, o, -r3),
            torch.cartesian_prod(r1, -r2, r3),
            torch.cartesian_prod(r1, -r2, o),
            torch.cartesian_prod(r1, -r2, -r3),
            torch.cartesian_prod(o, r2, r3),
            torch.cartesian_prod(o, r2, o),
            torch.cartesian_prod(o, r2, -r3),
            torch.cartesian_prod(o, o, r3),
        ]
    )


@torch.jit.script
def shifted_softplus(x):
    r"""Compute shifted soft-plus activation function.
    Modifed from 
    https://github.com/atomistic-machine-learning/schnetpack/tree/master/src/schnetpack/nn/activations.py
    .. math::
       y = \ln\left(1 + e^{-x}\right) - \ln(2)

    Args:
        x (torch.Tensor): input tensor.

    Returns:
        torch.Tensor: shifted soft-plus of input.

    """
    return functional.softplus(x) - torch.log(2.0)

@torch.jit.script
def return_x(x):
    """
    When not using activation, just return x
    """
    return x

@torch.jit.script
def gaussian_smearing(distances, offset, widths, centered: bool=False):
    r"""Smear interatomic distance values using Gaussian functions.

    Modified from original code found at 
    https://github.com/atomistic-machine-learning/schnetpack/blob/master/src/schnetpack/nn/acsf.py

    Args:
        distances (torch.Tensor): interatomic distances of (N_b x N_at x N_nbh) shape.
        offset (torch.Tensor): offsets values of Gaussian functions.
        widths: width values of Gaussian functions.
        centered (bool, optional): If True, Gaussians are centered at the origin and
            the offsets are used to as their widths (used e.g. for angular functions).

    Returns:
        torch.Tensor: smeared distances (N_b x N_at x N_nbh x N_g).

    """
    if not centered:
        # compute width of Gaussian functions (using an overlap of 1 STDDEV)
        coeff = -0.5 / torch.pow(widths, 2)
        # Use advanced indexing to compute the individual components
        diff = distances[:, :, :, None] - offset[None, None, None, :]
    else:
        # if Gaussian functions are centered, use offsets to compute widths
        coeff = -0.5 / torch.pow(offset, 2)
        # if Gaussian functions are centered, no offset is subtracted
        diff = distances[:, :, :, None]
    # compute smear distance values
    gauss = torch.exp(coeff * torch.pow(diff, 2))
    return gauss

@torch.jit.script
def atom_distances(
    positions,
    neighbors,
    cell,
    cell_offsets,
    neighbor_mask,
    return_vecs: Optional[bool]=False,
    normalize_vecs: Optional[bool]=False,
):
    r"""Compute distance of every atom to its neighbors.
    https://github.com/atomistic-machine-learning/schnetpack/blob/master/src/schnetpack/nn/neighbors.py

    This function uses advanced torch indexing to compute differentiable distances
    of every central atom to its relevant neighbors.

    Args:
        positions (torch.Tensor):
            atomic Cartesian coordinates with (N_b x N_at x 3) shape
        neighbors (torch.Tensor):
            indices of neighboring atoms to consider with (N_b x N_at x N_nbh) shape
        cell (torch.tensor, optional):
            periodic cell of (N_b x 3 x 3) shape
        cell_offsets (torch.Tensor, optional) :
            offset of atom in cell coordinates with (N_b x N_at x N_nbh x 3) shape
        return_vecs (bool, optional): if True, also returns direction vectors.
        normalize_vecs (bool, optional): if True, normalize direction vectors.
        neighbor_mask (torch.Tensor, optional): boolean mask for neighbor positions.

    Returns:
        (torch.Tensor, torch.Tensor):
            distances:
                distance of every atom to its neighbors with
                (N_b x N_at x N_nbh) shape.

            dist_vec:
                direction cosines of every atom to its
                neighbors with (N_b x N_at x N_nbh x 3) shape (optional).

    """

    # Construct auxiliary index vector
    n_batch = positions.size()[0]
    idx_m = torch.arange(n_batch, device=positions.device, dtype=torch.long)[
        :, None, None
    ]
    # Get atomic positions of all neighboring indices
    pos_xyz = positions[idx_m, neighbors[:, :, :], :]

    # Subtract positions of central atoms to get distance vectors
    dist_vec = pos_xyz - positions[:, :, None, :]

    # add cell offset
    if cell is not None:
        B, A, N, D = cell_offsets.size()
        cell_offsets = cell_offsets.view(B, A * N, D)
        offsets = cell_offsets.bmm(cell)
        offsets = offsets.view(B, A, N, D)
        dist_vec += offsets

    # Compute vector lengths
    distances = torch.norm(dist_vec, 2, 3)

    if neighbor_mask is not None:
        # Avoid problems with zero distances in forces (instability of square
        # root derivative at 0) This way is neccessary, as gradients do not
        # work with inplace operations, such as e.g.
        # -> distances[mask==0] = 0.0
        tmp_distances = torch.zeros_like(distances)
        tmp_distances[neighbor_mask != 0] = distances[neighbor_mask != 0]
        distances = tmp_distances

    #if return_vecs:
    #    tmp_distances = torch.ones_like(distances)
    #    tmp_distances[neighbor_mask != 0] = distances[neighbor_mask != 0]

    #    if normalize_vecs:
    #        dist_vec = dist_vec / tmp_distances[:, :, :, None]
    #    return distances, dist_vec
    return distances

class Dense(torch.nn.Linear):
    r"""Fully connected linear layer with activation function.
    Modified from original code found at 
    https://github.com/atomistic-machine-learning/schnetpack/blob/master/src/schnetpack/nn/base.py

    .. math::
       y = activation(xW^T + b)

    Args:
        in_features (int): number of input feature :math:`x`.
        out_features (int): number of output features :math:`y`.
        bias (bool, optional): if False, the layer will not adapt bias :math:`b`.
        activation (callable, optional): if None, no activation function is used.
        weight_init (callable, optional): weight initializer from current weight.
        bias_init (callable, optional): bias initializer from current bias.

    """

    def __init__(
        self,
        in_features,
        out_features,
        bias:bool=True,
        activation=shifted_softplus,
        weight_init=xavier_uniform_,
        bias_init=zeros_initializer,
        activate:bool=False
    ):
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.activate = activate

        if self.activate:
            self.activation = activation
        else:
            self.activation = return_x

        # initialize linear layer y = xW^T + b
        super(Dense, self).__init__(in_features, out_features, bias)

    def reset_parameters(self):
        """Reinitialize model weight and bias values."""
        self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, inputs: torch.Tensor):
        """Compute layer output.

        Args:
            inputs (dict of torch.Tensor): batch of input values.

        Returns:
            torch.Tensor: layer output.

        """
        # compute linear layer y = xW^T + b
        y = functional.linear(inputs, self.weight, self.bias)
        # add activation function
        if self.activate:
            y = self.activation(y)
        return y

class CFConv(nn.Module):
    r"""Continuous-filter convolution block used in SchNet module.
    Modified from original code found at 
    https://github.com/atomistic-machine-learning/schnetpack/blob/master/src/schnetpack/nn/cfconv.py

    Args:
        n_in (int): number of input (i.e. atomic embedding) dimensions.
        n_filters (int): number of filter dimensions.
        n_out (int): number of output dimensions.
        filter_network (nn.Module): filter block.
        cutoff_network (nn.Module, optional): if None, no cut off function is used.
        activation (callable, optional): if None, no activation function is used.
        normalize_filter (bool, optional): If True, normalize filter to the number
            of neighbors when aggregating.
        axis (int, optional): axis over which convolution should be applied.

    """

    def __init__(
        self,
        n_in,
        n_filters,
        n_out,
        filter_network,
        cutoff_network=None,
        activation=None,
        normalize_filter=False,
        axis=2,
    ):
        super(CFConv, self).__init__()
        self.in2f = Dense(n_in, n_filters, bias=False, activate=False)
        self.f2out = Dense(n_filters, n_out, bias=True, activate=True)
        self.filter_network = filter_network
        self.cutoff_network = cutoff_network
        self.agg = Aggregate(axis=axis, mean=normalize_filter)

    def forward(self, x, r_ij, neighbors, pairwise_mask, f_ij:Optional[torch.Tensor]=None):
        """Compute convolution block.

        Args:
            x (torch.Tensor): input representation/embedding of atomic environments
                with (N_b, N_a, n_in) shape.
            r_ij (torch.Tensor): interatomic distances of (N_b, N_a, N_nbh) shape.
            neighbors (torch.Tensor): indices of neighbors of (N_b, N_a, N_nbh) shape.
            pairwise_mask (torch.Tensor): mask to filter out non-existing neighbors
                introduced via padding.
            f_ij (torch.Tensor, optional): expanded interatomic distances in a basis.
                If None, r_ij.unsqueeze(-1) is used.

        Returns:
            torch.Tensor: block output with (N_b, N_a, n_out) shape.

        """
        if f_ij is None:
            f_ij = r_ij.unsqueeze(-1)

        # pass expanded interactomic distances through filter block
        W = self.filter_network(f_ij)
        # apply cutoff
        if self.cutoff_network is not None:
            C = self.cutoff_network(r_ij)
            W = W * C.unsqueeze(-1)

        # pass initial embeddings through Dense layer
        y = self.in2f(x)
        # reshape y for element-wise multiplication by W
        nbh_size = neighbors.size()
        nbh = neighbors.view(-1, nbh_size[1] * nbh_size[2], 1)
        nbh = nbh.expand(-1, -1, y.size(2))
        y = torch.gather(y, 1, nbh)
        y = y.view(nbh_size[0], nbh_size[1], nbh_size[2], -1)

        # element-wise multiplication, aggregating and Dense layer
        y = y * W
        y = self.agg(y, pairwise_mask)
        y = self.f2out(y)
        return y

class MLP(torch.nn.Module):
    """Multiple layer fully connected perceptron neural network.
    Modified from original code found at 
    https://github.com/atomistic-machine-learning/schnetpack/blob/master/src/schnetpack/nn/blocks.py

    Args:
        n_in (int): number of input nodes.
        n_out (int): number of output nodes.
        n_hidden (list of int or int, optional): number hidden layer nodes.
            If an integer, same number of node is used for all hidden layers resulting
            in a rectangular network.
            If None, the number of neurons is divided by two after each layer starting
            n_in resulting in a pyramidal network.
        n_layers (int, optional): number of layers.
        activation (callable, optional): activation function. All hidden layers would
            the same activation function except the output layer that does not apply
            any activation function.

    """

    def __init__(
        self, n_in, n_out, n_hidden=None, n_layers=2, activation=shifted_softplus
    ):
        super(MLP, self).__init__()
        # get list of number of nodes in input, hidden & output layers
        if n_hidden is None:
            c_neurons = n_in
            self.n_neurons = []
            for i in range(n_layers):
                self.n_neurons.append(c_neurons)
                c_neurons = c_neurons // 2
            self.n_neurons.append(n_out)
        else:
            # get list of number of nodes hidden layers
            if type(n_hidden) is int:
                n_hidden = [n_hidden] * (n_layers - 1)
            self.n_neurons = [n_in] + n_hidden + [n_out]

        # assign a Dense layer (with activation function) to each hidden layer
        layers = [
            Dense(self.n_neurons[i], self.n_neurons[i + 1], activate=True)
            for i in range(n_layers - 1)
        ]
        # assign a Dense layer (without activation function) to the output layer
        layers.append(Dense(self.n_neurons[-2], self.n_neurons[-1], activate=False))
        # put all layers together to make the network
        self.out_net = nn.Sequential(*layers)

    def forward(self, inputs):
        """Compute neural network output.

        Args:
            inputs (torch.Tensor): network input.

        Returns:
            torch.Tensor: network output.

        """
        return self.out_net(inputs)


class Aggregate(nn.Module):
    """Pooling layer based on sum or average with optional masking.
    Modified from original code found at
    https://github.com/atomistic-machine-learning/schnetpack/blob/master/src/schnetpack/nn/base.py

    Args:
        axis (int): axis along which pooling is done.
        mean (bool, optional): if True, use average instead for sum pooling.
        keepdim (bool, optional): whether the output tensor has dim retained or not.

    """

    def __init__(self, axis, mean=False, keepdim=True):
        super(Aggregate, self).__init__()
        self.average = mean
        self.axis = axis
        self.keepdim = keepdim

    def forward(self, input, mask:Optional[torch.Tensor]=None):
        r"""Compute layer output.

        Args:
            input (torch.Tensor): input data.
            mask (torch.Tensor, optional): mask to be applied; e.g. neighbors mask.

        Returns:
            torch.Tensor: layer output.

        """
        # mask input
        if mask is not None:
            input = input * mask[..., None]
        # compute sum of input along axis
        y = torch.sum(input, self.axis)
        # compute average of input along axis
        if self.average:
            # get the number of items along axis
            if mask is not None:
                N = torch.sum(mask, self.axis, keepdim=self.keepdim)
                N = torch.max(N, other=torch.ones_like(N))
            else:
                N = input.size(self.axis)
                N = torch.tensor(N, device=y.device)
            y = y / N
        return y

class AtomDistances(torch.nn.Module):
    r"""Layer for computing distance of every atom to its neighbors.
    Modified from original code found at 
    https://github.com/atomistic-machine-learning/schnetpack/blob/master/src/schnetpack/nn/neighbors.py

    Args:
        return_directions (bool, optional): if True, the `forward` method also returns
            normalized direction vectors.

    """

    def __init__(self, return_directions=False):
        super(AtomDistances, self).__init__()
        self.return_directions = return_directions

    def forward(
        self, positions, neighbors, cell, cell_offsets, neighbor_mask
    ):
        r"""Compute distance of every atom to its neighbors.

        Args:
            positions (torch.Tensor): atomic Cartesian coordinates with
                (N_b x N_at x 3) shape.
            neighbors (torch.Tensor): indices of neighboring atoms to consider
                with (N_b x N_at x N_nbh) shape.
            cell (torch.tensor, optional): periodic cell of (N_b x 3 x 3) shape.
            cell_offsets (torch.Tensor, optional): offset of atom in cell coordinates
                with (N_b x N_at x N_nbh x 3) shape.
            neighbor_mask (torch.Tensor, optional): boolean mask for neighbor
                positions. Required for the stable computation of forces in
                molecules with different sizes.

        Returns:
            torch.Tensor: layer output of (N_b x N_at x N_nbh) shape.

        """
        return atom_distances(
            positions,
            neighbors,
            cell,
            cell_offsets,
            neighbor_mask,
            return_vecs=self.return_directions,
            normalize_vecs=True,
        )

class GaussianSmearing(torch.nn.Module):
    r"""Smear layer using a set of Gaussian functions.
    Modified from original code found at 
    https://github.com/atomistic-machine-learning/schnetpack/blob/master/src/schnetpack/nn/acsf.py

    Args:
        start (float, optional): center of first Gaussian function, :math:`\mu_0`.
        stop (float, optional): center of last Gaussian function, :math:`\mu_{N_g}`
        n_gaussians (int, optional): total number of Gaussian functions, :math:`N_g`.
        centered (bool, optional): If True, Gaussians are centered at the origin and
            the offsets are used to as their widths (used e.g. for angular functions).
        trainable (bool, optional): If True, widths and offset of Gaussian functions
            are adjusted during training process.

    """

    def __init__(
        self, start=0.0, stop=5.0, n_gaussians=50, centered: Optional[bool]=False, trainable: Optional[bool]=False
    ):
        super(GaussianSmearing, self).__init__()
        # compute offset and width of Gaussian functions
        offset = torch.linspace(start, stop, n_gaussians)
        widths = torch.FloatTensor((offset[1] - offset[0]) * torch.ones_like(offset))
        if trainable:
            self.width = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer("width", widths)
            self.register_buffer("offsets", offset)
        self.centered = centered

    def forward(self, distances):
        """Compute smeared-gaussian distance values.

        Args:
            distances (torch.Tensor): interatomic distance values of
                (N_b x N_at x N_nbh) shape.

        Returns:
            torch.Tensor: layer output of (N_b x N_at x N_nbh x N_g) shape.

        """
        return gaussian_smearing(
            distances, self.offsets, self.width, centered=self.centered
        )

class GetItem(torch.nn.Module):
    """Extraction layer to get an item from SchNetPack dictionary of input tensors.
    This doesn't work with nn.Sequence in TorchScript, so this now returns the inputs,
    which is already the Property to be extracted

    Modified from original code found at 
    https://github.com/atomistic-machine-learning/schnetpack/blob/master/src/schnetpack/nn/base.py

    Args:
        key (str): Property to be extracted from SchNetPack input tensors.

    """

    def __init__(self, key: str):
        super(GetItem, self).__init__()
        self.key = key

    def forward(self, inputs):
        """Compute layer output.

        Args:
            inputs (tensor of torch.Tensor): SchNetPack input tensor.

        Returns:
            torch.Tensor: layer output.

        """
        return inputs

