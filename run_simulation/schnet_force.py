import torch
import sys
from schnetpack import Properties 
import torch.nn as nn
import schnetpack as sch
from typing import Optional, Dict
from schnetpack.nn.acsf import gaussian_smearing
from schnetpack.nn.activations import shifted_softplus
from schnetpack.nn.cfconv import CFConv
from torch.nn import functional
from torch.nn.init import xavier_uniform_
from schnetpack.nn.initializers import zeros_initializer
from nn_classes import *
import numpy as np

class SchNetForce(torch.nn.Module):
    """
    SchNetForce used for getting energies and forces from SchNet in OpenMM
    This version of the class does not use periodic boundary conditions
    """
    def __init__(self, atomic_numbers, nn_index, representation, output_modules, device='cuda'):
        """
        atomic_numbers : list 
            List of atomic numbers for each atom that the neural network is applied to.
            Should be same len as nn_index
        nn_index : list
            List of atom indices that the neural network is applied to. It is used to slice
            the full positions tensor passed in by OpenMM.
        representation : torch.nn.Module
            class that computes the representation (in this case SchNet) of the system
        output_modules : torch.nn.Module
            class that computes the energy for a given representation
        device : str
            device which torch will be run on. Default is cuda
        """
        super(SchNetForce, self).__init__()
        self.device = torch.device(device)
        self.atomic_numbers = torch.tensor(atomic_numbers, device=self.device)
        self.nn_index = nn_index
        if len(self.atomic_numbers) != len(self.nn_index):
            print("WARNING: The length of the list of atomic numbers should be equal to the length of the list of \
                    nn_indices")
        #Since PBCs not being used, cell can be defined as the same each iteration
        self.cell = torch.zeros((3,3), device=self.device)

        #GetEnvironmentSimple returns a "neighbor list" of each atom in the structure
        self.environment = GetEnvironmentSimple(device=device)
        self.representation = representation
   
        if type(output_modules) not in [list, torch.nn.ModuleList]:
            output_modules = [output_modules]
        if type(output_modules) == list:
            output_modules = torch.nn.ModuleList(output_modules)
        self.output_modules = output_modules

    def forward(self, positions):
        """The forward method returns the energy computed from positions.

        Parameters
        ----------
        positions : torch.Tensor with shape (nparticles,3)
           positions[i,k] is the position (in nanometers) of spatial dimension k of particle i

        Returns
        -------
        potential : torch.Scalar
           The potential energy (in kJ/mol)
        """
        inputs = {}
        #Convert positions to angstrom from nm
        positions = positions * 10
        #Slice total positions tensor with self.nn_index
        nn_positions = positions[self.nn_index]
        #Create neural network inputs
        #INFO environment stays the same in the simple provider, so can probably change
        #get environment so it doesn't get called every frame
        neighborhood_idx, offsets = self.environment.get_environment(nn_positions)
        inputs['_atomic_numbers'] = self.atomic_numbers.unsqueeze(0)
        inputs['_atom_mask'] = torch.ones_like(inputs['_atomic_numbers']).float()
        inputs['_positions'] = nn_positions.unsqueeze(0)
        inputs['_neighbors'] = neighborhood_idx.unsqueeze(0).long()
        inputs['_cell'] = self.cell.unsqueeze(0)
        inputs['_cell_offset'] = offsets.unsqueeze(0)

        # Calculate masks
        inputs['_neighbor_mask'] = torch.ones_like(inputs['_atomic_numbers'], device=self.device).float()
        mask = inputs['_neighbors'] >= 0
        inputs['_neighbor_mask'] = mask.float()
        inputs['_neighbors'] = (
            inputs['_neighbors'] * inputs['_neighbor_mask'].long()
        )
        
        inputs["representation"] = self.representation(inputs)
        for out in self.output_modules:
            y = out(inputs)
        return y

class SchNetForcePBC(torch.nn.Module):
    """
    SchNetForce used for getting energies and forces from SchNet in OpenMM
    This version of the class does use periodic boundary conditions
    """
    def __init__(self, atomic_numbers, nn_index, representation, output_modules, cutoff=5, device='cuda'):
        """
        atomic_numbers : list
            List of atomic numbers for each atom that the neural network is applied to.
            Should be same len as nn_index
        nn_index : list
            List of atom indices that the neural network is applied to. It is used to slice
            the full positions tensor passed in by OpenMM.
        representation : torch.nn.Module
            class that computes the representation (in this case SchNet) of the system
        output_modules : torch.nn.Module
            class that computes the energy for a given representation
        cutoff : float
            Neural netowrk cutoff used for gathering atom neighbors in GetEnvironmentTorch
        device : str
            device which torch will be run on. Default is cuda
        """
        super(SchNetForcePBC, self).__init__()
        self.device = torch.device(device)
        self.atomic_numbers = torch.tensor(atomic_numbers, device=self.device)
        self.nn_index = nn_index
        self.environment = GetEnvironmentTorch(cutoff, device=device)
        self.representation = representation

        if type(output_modules) not in [list, torch.nn.ModuleList]:
            output_modules = [output_modules]
        if type(output_modules) == list:
            output_modules = torch.nn.ModuleList(output_modules)
        self.output_modules = output_modules

    def forward(self, positions, boxvectors):
        """The forward method returns the energy computed from positions.

        Parameters
        ----------
        positions : torch.Tensor with shape (nparticles,3)
           positions[i,k] is the position (in nanometers) of spatial dimension k of particle i
        boxvectors : torch.Tensor with shape (3,3)
            boxvectors containing the cell dimensions of the system

        Returns
        -------
        potential : torch.Scalar
           The potential energy (in kJ/mol)
        """
        inputs = {}
        #Convert both positions and boxvectors to angstrom
        positions = positions * 10
        boxvectors = boxvectors * 10
        nn_positions = positions[self.nn_index]
        #TO DO: change so get_environment isn't called every step
        neighborhood_idx, offsets = self.environment.get_environment(nn_positions, boxvectors, self.atomic_numbers)
        inputs['_atomic_numbers'] = self.atomic_numbers.unsqueeze(0)
        inputs['_atom_mask'] = torch.ones_like(inputs['_atomic_numbers']).float()
        inputs['_positions'] = nn_positions.unsqueeze(0)
        inputs['_neighbors'] = neighborhood_idx.unsqueeze(0).long()
        inputs['_cell'] = boxvectors.unsqueeze(0) 
        inputs['_cell_offset'] = offsets.unsqueeze(0)

        # Calculate masks
        inputs['_neighbor_mask'] = torch.ones_like(inputs['_atomic_numbers'], device=self.device).float()
        mask = inputs['_neighbors'] >= 0
        inputs['_neighbor_mask'] = mask.float()
        inputs['_neighbors'] = (
            inputs['_neighbors'] * inputs['_neighbor_mask'].long()
        )

        inputs["representation"] = self.representation(inputs)
        for out in self.output_modules:
            y = out(inputs)
        return y

class SchNetInteraction(torch.nn.Module):
    r"""SchNet interaction block for modeling interactions of atomistic systems.
    Modified from code found at 
    https://github.com/atomistic-machine-learning/schnetpack/tree/master/src/schnetpack/representation/schnet.py

    Args:
        n_atom_basis (int): number of features to describe atomic environments.
        n_spatial_basis (int): number of input features of filter-generating networks.
        n_filters (int): number of filters used in continuous-filter convolution.
        cutoff (float): cutoff radius.
        cutoff_network (nn.Module, optional): cutoff layer.
        normalize_filter (bool, optional): if True, divide aggregated filter by number
            of neighbors over which convolution is applied.

    """

    def __init__(
        self,
        n_atom_basis,
        n_spatial_basis,
        n_filters,
        cutoff,
        cutoff_network=sch.nn.cutoff.CosineCutoff,
        normalize_filter:bool=False,
    ):
        super(SchNetInteraction, self).__init__()
        # filter block used in interaction block
        self.filter_network = nn.Sequential(
            Dense(n_spatial_basis, n_filters, activate=True),
            Dense(n_filters, n_filters, activate=False),
        )
        # cutoff layer used in interaction block
        self.cutoff_network = cutoff_network(cutoff)
        # interaction block
        self.cfconv = CFConv(
            n_atom_basis,
            n_filters,
            n_atom_basis,
            self.filter_network,
            cutoff_network=self.cutoff_network,
            activation=shifted_softplus,
            normalize_filter=normalize_filter,
        )
        # dense layer
        self.dense = Dense(n_atom_basis, n_atom_basis, bias=True, activate=False)

    def forward(self, x, r_ij, neighbors, neighbor_mask, f_ij:Optional[torch.Tensor]=None):
        """Compute interaction output.

        Args:
            x (torch.Tensor): input representation/embedding of atomic environments
                with (N_b, N_a, n_atom_basis) shape.
            r_ij (torch.Tensor): interatomic distances of (N_b, N_a, N_nbh) shape.
            neighbors (torch.Tensor): indices of neighbors of (N_b, N_a, N_nbh) shape.
            neighbor_mask (torch.Tensor): mask to filter out non-existing neighbors
                introduced via padding.
            f_ij (torch.Tensor, optional): expanded interatomic distances in a basis.
                If None, r_ij.unsqueeze(-1) is used.

        Returns:
            torch.Tensor: block output with (N_b, N_a, n_atom_basis) shape.

        """
        # continuous-filter convolution interaction block followed by Dense layer
        v = self.cfconv(x, r_ij, neighbors, neighbor_mask, f_ij)
        v = self.dense(v)
        return v

class SchNet(torch.nn.Module):
    """SchNet architecture for learning representations of atomistic systems.
    Modifed from original code found at 
    https://github.com/atomistic-machine-learning/schnetpack/tree/master/src/schnetpack/representation/schnet.py

    Args:
        n_atom_basis (int, optional): number of features to describe atomic environments.
            This determines the size of each embedding vector; i.e. embeddings_dim.
        n_filters (int, optional): number of filters used in continuous-filter convolution
        n_interactions (int, optional): number of interaction blocks.
        cutoff (float, optional): cutoff radius.
        n_gaussians (int, optional): number of Gaussian functions used to expand
            atomic distances.
        normalize_filter (bool, optional): if True, divide aggregated filter by number
            of neighbors over which convolution is applied.
        coupled_interactions (bool, optional): if True, share the weights across
            interaction blocks and filter-generating networks.
        return_intermediate (bool, optional): if True, `forward` method also returns
            intermediate atomic representations after each interaction block is applied.
        max_z (int, optional): maximum nuclear charge allowed in database. This
            determines the size of the dictionary of embedding; i.e. num_embeddings.
        cutoff_network (nn.Module, optional): cutoff layer.
        trainable_gaussians (bool, optional): If True, widths and offset of Gaussian
            functions are adjusted during training process.
        distance_expansion (nn.Module, optional): layer for expanding interatomic
            distances in a basis.
        charged_systems (bool, optional):

    References:
    .. [#schnet1] Schütt, Arbabzadah, Chmiela, Müller, Tkatchenko:
       Quantum-chemical insights from deep tensor neural networks.
       Nature Communications, 8, 13890. 2017.
    .. [#schnet_transfer] Schütt, Kindermans, Sauceda, Chmiela, Tkatchenko, Müller:
       SchNet: A continuous-filter convolutional neural network for modeling quantum
       interactions.
       In Advances in Neural Information Processing Systems, pp. 992-1002. 2017.
    .. [#schnet3] Schütt, Sauceda, Kindermans, Tkatchenko, Müller:
       SchNet - a deep learning architecture for molceules and materials.
       The Journal of Chemical Physics 148 (24), 241722. 2018.

    """

    def __init__(
        self,
        n_atom_basis=128,
        n_filters=128,
        n_interactions=3,
        cutoff=5.0,
        n_gaussians=25,
        normalize_filter: bool=False,
        coupled_interactions: bool=False,
        return_intermediate: bool=False,
        max_z=100,
        cutoff_network=sch.nn.CosineCutoff,
        trainable_gaussians: bool=False,
        distance_expansion: Optional[torch.Tensor]=None,
        charged_systems: bool=False,
    ):
        super(SchNet, self).__init__()

        self.n_atom_basis = n_atom_basis
        # make a lookup table to store embeddings for each element (up to atomic
        # number max_z) each of which is a vector of size n_atom_basis
        self.embedding = nn.Embedding(max_z, n_atom_basis, padding_idx=0)

        # layer for computing interatomic distances
        self.distances = AtomDistances()

        # layer for expanding interatomic distances in a basis
        if distance_expansion is None:
            self.distance_expansion = GaussianSmearing(
                0.0, cutoff, n_gaussians, trainable=trainable_gaussians
            )
        else:
            self.distance_expansion = distance_expansion

        # block for computing interaction
        if coupled_interactions:
            # use the same SchNetInteraction instance (hence the same weights)
            self.interactions = nn.ModuleList(
                [
                    SchNetInteraction(
                        n_atom_basis=n_atom_basis,
                        n_spatial_basis=n_gaussians,
                        n_filters=n_filters,
                        cutoff_network=cutoff_network,
                        cutoff=cutoff,
                        normalize_filter=normalize_filter,
                    )
                ]
                * n_interactions
            )
        else:
            # use one SchNetInteraction instance for each interaction
            self.interactions = nn.ModuleList(
                [
                    SchNetInteraction(
                        n_atom_basis=n_atom_basis,
                        n_spatial_basis=n_gaussians,
                        n_filters=n_filters,
                        cutoff_network=cutoff_network,
                        cutoff=cutoff,
                        normalize_filter=normalize_filter,
                    )
                    for _ in range(n_interactions)
                ]
            )

        # set attributes
        self.return_intermediate = return_intermediate
        self.charged_systems = charged_systems
        if charged_systems:
            self.charge = nn.Parameter(torch.Tensor(1, n_atom_basis))
            self.charge.data.normal_(0, 1.0 / n_atom_basis ** 0.5)

    def forward(self, inputs:Dict[str, torch.Tensor]):
        """Compute atomic representations/embeddings.

        Args:
            inputs (dict of torch.Tensor): SchNetPack dictionary of input tensors.

        Returns:
            torch.Tensor: atom-wise representation.
            list of torch.Tensor: intermediate atom-wise representations, if
            return_intermediate=True was used.

        """
        # get tensors from input dictionary
        atomic_numbers = inputs["_atomic_numbers"]
        positions = inputs["_positions"]
        cell = inputs["_cell"]
        cell_offset = inputs["_cell_offset"]
        neighbors = inputs["_neighbors"]
        neighbor_mask = inputs["_neighbor_mask"]
        atom_mask = inputs["_atom_mask"]

        # get atom embeddings for the input atomic numbers
        x = self.embedding(atomic_numbers)
        
        # compute interatomic distance of every atom to its neighbors
        r_ij = self.distances(
            positions, neighbors, cell, cell_offset, neighbor_mask
        )
        # expand interatomic distances (for example, Gaussian smearing)
        f_ij = self.distance_expansion(r_ij)
        # compute interaction block to update atomic embeddings
        for interaction in self.interactions:
            v = interaction(x, r_ij, neighbors, neighbor_mask, f_ij=f_ij)
            x = x + v

        return x

class GetEnvironmentSimple(torch.nn.Module):
    """
    Returns the indices of all atoms for each atom in the system
    Serves as a neighbor list. Does not work with cutoffs
    Modified from the original code found at
    https://github.com/atomistic-machine-learning/schnetpack/tree/master/src/schnetpack/environment.py
    """
    def __init__(self, device):
        """
        Args
        device: str
            str listing the device for torch to run on
        """

        super(GetEnvironmentSimple, self).__init__()
        self.device = torch.device(device)

    def get_environment(self, positions):
        """
        Computes environment for each atom

        Args
        positions : torch.Tensor(n_atoms, 3)
            tensor containing the positions of each atom

        Returns 
        neighborood_idx : torch.tensor(n_atoms, n_atoms-1)
            First dimension corresponds to all atoms in the geometry. Second dimension contains the 
            indices of all other atoms (works as a neighbor list)
        offsets : torch.tensor(n_atoms, n_atoms-1, 3)
            Normally, this would contain the number of cell vectors separating one atom to another
            in each dimension. Since PBCs aren't being used here, it is a tensor of all zeros.
        """

        n_atoms = positions.shape[0]
        if n_atoms == 1:
            neighborhood_idx = -torch.ones((1, 1), dtype=torch.float, device=self.device)
            offsets = torch.zeros((n_atoms, 1, 3), dtype=torch.float, device=self.device)
        else:
            neighborhood_idx = torch.tile(
                torch.arange(n_atoms, dtype=torch.float, device=self.device)[None], (n_atoms, 1)
            )

            neighborhood_idx = neighborhood_idx[
                ~torch.eye(n_atoms, dtype=torch.bool, device=self.device)
            ].reshape(n_atoms, n_atoms - 1)

            offsets = torch.zeros(
                (neighborhood_idx.shape[0], neighborhood_idx.shape[1], 3),
                dtype=torch.float, device=self.device
            )

        return neighborhood_idx, offsets

class GetEnvironmentTorch(torch.nn.Module):
    """
    Returns the indices of the neighboring atoms for each atom.
    Works with a cutoff and PBCs
    Modified from the original code found at
    https://github.com/atomistic-machine-learning/schnetpack/tree/master/src/schnetpack/environment.py
    """
    def __init__(self, cutoff, device):
        """
        Args
        cutoff : float
            GetEnvironmentTorch returns all atoms within a cutoff (assuming periodic boundary conditions)
        device: str
            str listing the device for torch to run on
        """
        super(GetEnvironmentTorch, self).__init__()
        self.cutoff = cutoff
        self.device = torch.device(device)

    def get_environment(self, positions, boxvectors, species):
        """
        Computes environment for each atom

        Args
        positions : torch.Tensor(n_atoms, 3)
            tensor containing the positions of each atom
        boxvectors : torch.Tensor(3,3)
            tensor containing the cell vectors
        species : torch.Tensor(n_atoms)
            contains atomic numbers

        Returns
        neighborood_idx : torch.Tensor
            First dimension corresponds to all atoms in the geometry. Second dimension contains the
            indices of neighboring atoms
        offsets : torch.Tensor
            Tensor containing the number of cell lengths separating atom in first dimension from atom in 
            second dimension
        """

        
        shifts = compute_shifts(cell=boxvectors, pbc=torch.tensor([True, True, True], device=self.device), cutoff=self.cutoff)

        idx_i, idx_j, idx_S = neighbor_pairs(
            species == -1, positions, boxvectors, shifts, self.cutoff
        )

        # Create bidirectional id arrays, similar to what the ASE neighbor_list returns
        bi_idx_i = torch.hstack((idx_i, idx_j))
        bi_idx_j = torch.hstack((idx_j, idx_i))
        bi_idx_S = torch.vstack((-idx_S, idx_S)).float()

        n_atoms = positions.shape[0]

        if bi_idx_i.shape[0] > 0:
            uidx, n_nbh = torch.unique(bi_idx_i, return_counts=True)
            n_max_nbh = torch.max(n_nbh)
            
            n_nbh = torch.tile(n_nbh[:, None], (1, int(n_max_nbh))).to(self.device)
            nbh_range = torch.tile(
                torch.arange(n_max_nbh, device=self.device)[None], (n_nbh.shape[0], 1),
            )

            mask = torch.zeros((n_atoms, int(torch.max(n_max_nbh))), dtype=torch.bool, device=self.device)
            mask[uidx, :] = nbh_range < n_nbh
            neighborhood_idx = -torch.ones((n_atoms, int(torch.max(n_max_nbh))), dtype=torch.long, device=self.device)
            offset = torch.zeros((n_atoms, int(torch.max(n_max_nbh)), 3), dtype=torch.float, device=self.device)

            # Assign neighbors and offsets according to the indices in bi_idx_i, since in contrast
            # to the ASE provider the bidirectional arrays are no longer sorted.
            # TODO: There might be a more efficient way of doing this than a loop
            for idx in range(n_atoms):
                neighborhood_idx[idx, mask[idx]] = bi_idx_j[bi_idx_i == idx]
                offset[idx, mask[idx]] = bi_idx_S[bi_idx_i == idx]

        else:
            neighborhood_idx = -torch.ones((n_atoms, 1), dtype=torch.float)
            offset = torch.zeros((n_atoms, 1, 3), dtype=torch.float)

        return neighborhood_idx, offset

class Atomwise(torch.nn.Module):
    """
    Predicts atom-wise contributions and accumulates global prediction, e.g. for the
    energy.
    Modified from code at
    https://github.com/atomistic-machine-learning/schnetpack/tree/master/src/schnetpack/atomistic/output_modules.py

    Args:
        n_in (int): input dimension of representation
        n_out (int): output dimension of target property (default: 1)
        aggregation_mode (str): one of {sum, avg} (default: sum)
        n_layers (int): number of nn in output network (default: 2)
        n_neurons (list of int or None): number of neurons in each layer of the output
            network. If `None`, divide neurons by 2 in each layer. (default: None)
        activation (function): activation function for hidden nn
            (default: spk.nn.activations.shifted_softplus)
        property (str): name of the output property (default: "y")
        contributions (str or None): Name of property contributions in return dict.
            No contributions returned if None. (default: None)
        derivative (str or None): Name of property derivative. No derivative
            returned if None. (default: None)
        negative_dr (bool): Multiply the derivative with -1 if True. (default: False)
        stress (str or None): Name of stress property. Compute the derivative with
            respect to the cell parameters if not None. (default: None)
        create_graph (bool): If False, the graph used to compute the grad will be
            freed. Note that in nearly all cases setting this option to True is not nee
            ded and often can be worked around in a much more efficient way. Defaults to
            the value of create_graph. (default: False)
        mean (torch.Tensor or None): mean of property
        stddev (torch.Tensor or None): standard deviation of property (default: None)
        atomref (torch.Tensor or None): reference single-atom properties. Expects
            an (max_z + 1) x 1 array where atomref[Z] corresponds to the reference
            property of element Z. The value of atomref[0] must be zero, as this
            corresponds to the reference property for for "mask" atoms. (default: None)
        outnet (callable): Network used for atomistic outputs. Takes schnetpack input
            dictionary as input. Output is not normalized. If set to None,
            a pyramidal network is generated automatically. (default: None)

    Returns:
        tuple: prediction for property

        If contributions is not None additionally returns atom-wise contributions.

        If derivative is not None additionally returns derivative w.r.t. atom positions.

    """

    def __init__(
        self,
        n_in,
        n_out=1,
        aggregation_mode="sum",
        n_layers=2,
        n_neurons=None,
        activation=shifted_softplus,
        property="y",
        outnet=None,
        mean=None,
        stddev=None,
    ):
        super(Atomwise, self).__init__()

        self.n_layers = n_layers
        self.property = property

        # build output network
        if outnet is None:
            self.out_net = nn.Sequential(
                GetItem("representation"),
                MLP(n_in, n_out, n_neurons, n_layers, activation),
            )
        else:
            self.out_net = outnet

        mean = torch.FloatTensor([0.0]) if mean is None else mean
        stddev = torch.FloatTensor([1.0]) if stddev is None else stddev

        self.standardize = sch.nn.base.ScaleShift(mean, stddev)

        # build aggregation layer
        if aggregation_mode == "sum":
            self.atom_pool = Aggregate(axis=1, mean=False)
        elif aggregation_mode == "avg":
            self.atom_pool = Aggregate(axis=1, mean=True)
        else:
            raise AtomwiseError(
                "{} is not a valid aggregation " "mode!".format(aggregation_mode)
            )

    def forward(self, inputs: Dict[str, torch.Tensor]):
        r"""
        predicts atomwise property
        """
        atomic_numbers = inputs['_atomic_numbers']
        atom_mask = inputs['_atom_mask']

        input_dict = inputs['representation']
        # run prediction
        yi = self.out_net(input_dict)
        yi = self.standardize(yi)

        y = self.atom_pool(yi, atom_mask)
        return y
