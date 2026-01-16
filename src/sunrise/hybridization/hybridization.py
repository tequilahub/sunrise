import numpy as np
from copy import deepcopy
from .utils import HybridizationUtils,AtomUtils
from .atom import Atom
BOND_DISTANCE_TOLERANCE = 1.3

class Graph:
    def __init__(self, atoms):
        self.atoms = atoms
        self.atom_indices = {atom: index for index, atom in enumerate(atoms)}
        self.bonds = {atom: [] for atom in atoms}
        self.bond_matrix = np.zeros((len(atoms), len(atoms)), dtype=int)
        self.__create_bonds()

    def __create_bonds(self):
        num_atoms = len(self.atoms)
        positions = np.array([atom.coords for atom in self.atoms])
        radii = np.array([atom.covalent_radius for atom in self.atoms])
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=2))
        bonding_threshold = BOND_DISTANCE_TOLERANCE * (
                radii[:, np.newaxis] + radii[np.newaxis, :]
        )
        bond_matrix = distances <= bonding_threshold

        # Track current bond counts for each atom to ensure we respect max_bonds:
        bond_counts = {atom: 0 for atom in self.atoms}
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                if bond_matrix[i, j]:
                    # Only form a bond if both atoms can still accept more bonds.
                    if (bond_counts[self.atoms[i]] < self.atoms[i].max_bonds and
                            bond_counts[self.atoms[j]] < self.atoms[j].max_bonds):
                        self.bonds[self.atoms[i]].append(self.atoms[j])
                        self.bonds[self.atoms[j]].append(self.atoms[i])
                        self.bond_matrix[i][j] = self.bond_matrix[j][i] = 1

                        bond_counts[self.atoms[i]] += 1
                        bond_counts[self.atoms[j]] += 1
        # Next, increase the bond multiplicity wherever possible.
        for i in range(num_atoms):
            #Preference in the bond multiplicity by smaller bonding, otherwise it would  be preference by order on the Zmatrix
            bonded = {j:distances[i,j] for j in range(num_atoms) if self.bond_matrix[i][j] > 0}
            bonded = dict(sorted(bonded.items(), key=lambda item: item[1]))
            for j in bonded:
                # Check if there's already a bond between atoms i and j
                while (bond_counts[self.atoms[i]] < self.atoms[i].max_bonds and bond_counts[self.atoms[j]] < self.atoms[j].max_bonds):
                    # Increase the bond multiplicity
                    self.bond_matrix[i][j] += 1
                    self.bond_matrix[j][i] += 1
                    bond_counts[self.atoms[i]] += 1
                    bond_counts[self.atoms[j]] += 1
        # Now bond all remaining atoms, prioritizing those with the most missing bonds
        offset_distances = distances - bonding_threshold

        def get_remaining_bonds(atom):
            return atom.max_bonds - bond_counts[atom]

        atoms_sorted = sorted(self.atoms, key=get_remaining_bonds, reverse=True)
        for x in range(num_atoms):
            i = self.atom_indices[atoms_sorted[x]]
            for k in range(get_remaining_bonds(atoms_sorted[x])):
                best = (None, float('inf'))
                for y in range(x + 1, num_atoms):
                    j = self.atom_indices[atoms_sorted[y]]
                    if not self.bond_matrix[i][j] and bond_counts[self.atoms[j]] < self.atoms[j].max_bonds and \
                            offset_distances[i][j] < best[1]:
                        best = (j, offset_distances[i][j])
                j = best[0]
                if j is not None:
                    self.bonds[self.atoms[i]].append(self.atoms[j])
                    self.bonds[self.atoms[j]].append(self.atoms[i])
                    self.bond_matrix[i][j] = self.bond_matrix[j][i] = 1

                    bond_counts[self.atoms[i]] += 1
                    bond_counts[self.atoms[j]] += 1
                else:
                    break

    def is_bonded(self, atom1, atom2):
        return self.bond_matrix[self.atom_indices[atom1]][self.atom_indices[atom2]] != 0

    def get_bonds(self, atom)->(Atom):
        return self.bonds.get(atom, [])

    def get_bond_multiplicity(self, atom1, atom2)->int:
        return self.bond_matrix[self.atom_indices[atom1]][self.atom_indices[atom2]]

    def get_multiplied_bonds(self, atom)->(Atom):
        """
        Returns a list of bonds for the given atom. For multiple bonds between two atoms,
        it returns the bond multiple times based on the bond multiplicity.
        """
        #NOTE: This commented version is original Steinhauser code. It returs [X0,X1,X1,X2...] while mine returns [X0,X1,X2,...,X1,...]


        # multiplied_bonds = []
        # atom_idx = self.atom_indices[atom]

        # for bonded_atom in self.get_bonds(atom):
        #     bonded_atom_idx = self.atom_indices[bonded_atom]
        #     bond_count = self.bond_matrix[atom_idx][bonded_atom_idx]

        #     for _ in range(bond_count):
        #         multiplied_bonds.append(bonded_atom)

        multiplied_bonds = [*self.get_bonds(atom)]
        for bonded in self.get_bonds(atom=atom):
            multi = self.get_bond_multiplicity(atom,bonded)
            if multi > 1:
                multiplied_bonds += (multi-1)*[bonded]
        return multiplied_bonds

    @staticmethod
    def parse_xyz(xyz_string, rng=None):
        atoms = Atom.parse_xyz(xyz_string)
        return Graph(atoms)

    def get_indexed_edges(self):
        edges = []
        visited = set()

        for i, atom in enumerate(self.atoms):
            for bonded_atom in self.get_bonds(atom):
                j = self.atom_indices[bonded_atom]

                # Order bond indices as (min, max) to avoid [i,j] and [j,i] duplication.
                bond = tuple(sorted([i, j]))

                if bond not in visited:
                    edges.append(list(bond))  # Add the bond as [i, j]
                    visited.add(bond)

        return edges

    def get_atom_indices(self,atom)->int:
        return self.atom_indices[atom]

    @staticmethod
    def get_angle(atom, neighbor1, neighbor2):
        vec1 = neighbor1.coords - atom.coords
        vec2 = neighbor2.coords - atom.coords
        cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        # Constrain result to avoid floating point errors (cos_theta can accidentally be slightly out of bounds)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return np.degrees(np.arccos(cos_theta))

    def get_angles(self, atom):
        bonded_atoms = self.get_bonds(atom)
        num_bonds = len(bonded_atoms)
        return [(self.get_angle(atom, bonded_atoms[i], bonded_atoms[j]))
                for i in range(num_bonds) for j in range(i + 1, num_bonds)]

    def get_angle_data(self, atom):
        bonded_atoms = self.get_bonds(atom)
        num_bonds = len(bonded_atoms)
        return [(self.get_angle(atom, bonded_atoms[i], bonded_atoms[j]), bonded_atoms[i], bonded_atoms[j])
                for i in range(num_bonds) for j in
                range(i + 1, num_bonds)]  # List of tuples: (Angle, BondedAtom1, BondedAtom2)

    def get_vectors(self, atom):
        bonded_atoms = self.get_bonds(atom)
        atom_coords = atom.coords
        return [bonded_atom.coords - atom_coords for bonded_atom in bonded_atoms]

    def get_vector_data(self, atom):
        bonded_atoms = self.get_bonds(atom)
        atom_coords = atom.coords
        return [(bonded_atom.coords - atom_coords, bonded_atom) for bonded_atom in bonded_atoms]

    def get_hybridization_data(self, atom):
        angle_data = self.get_angle_data(atom)
        num_bonds = len(self.get_bonds(atom))

        # Check if enough bonds for hybridization calculation
        if num_bonds == 0:
            return None, None
        elif num_bonds == 1:
            return 1, None

        # Find all hits for each hybridization angle range (±10 degrees tolerance)
        hit_data = {1: [], 2: [], 3: []}
        for angle, neighbor1, neighbor2 in angle_data:
            if 170 <= angle <= 190:  # sp (180°)
                hit_data[1].append((angle, neighbor1, neighbor2))
            elif 110 <= angle <= 130:  # sp² (120°)
                hit_data[2].append((angle, neighbor1, neighbor2))
            elif 99.5 <= angle <= 119.5:  # sp³ (109.5°)
                hit_data[3].append((angle, neighbor1, neighbor2))

        # Determine the best hybridization based on number of hits
        best_hybridization = max(hit_data, key=lambda sp: len(hit_data[sp]))
        if len(hit_data[best_hybridization]) == 0:
            return min(3, num_bonds - 1), None

        return best_hybridization, hit_data

    def get_hybridization(self, atom):
        hybridization, _ = self.get_hybridization_data(atom)
        return hybridization

    @staticmethod
    def get_hit_data(angle_data, sp):
        if sp == 1:  # For sp1 (180°)
            return [(angle, neighbor1, neighbor2) for angle, neighbor1, neighbor2 in angle_data if 170 <= angle <= 190]
        elif sp == 2:  # For sp² (120°)
            return [(angle, neighbor1, neighbor2) for angle, neighbor1, neighbor2 in angle_data if 110 <= angle <= 130]
        elif sp == 3:  # For sp³ (109.5°)
            return [(angle, neighbor1, neighbor2) for angle, neighbor1, neighbor2 in angle_data if
                    99.5 <= angle <= 119.5]
        else:
            return None

    @staticmethod
    def hybridization_orbital_vectors(sp):
        """
        Get the standard hybrid orbital directions for an atom based on its hybridization type.

        :param sp: The hybridization type (1 = sp, 2 = sp2, 3 = sp3)
        :return: A matrix where each row corresponds to a hybridized orbital direction.
        """
        if sp == 3:
            # sp³ is tetrahedral, bond angles ~ 109.5°
            return np.array([
                [1, 1, 1],
                [-1, 1, 1],
                [1, -1, 1],
                [1, 1, -1]
            ]) / np.sqrt(3)  # Normalize the vectors

        elif sp == 2:
            # sp² is trigonal planar, bond angles 120°
            return np.array([
                [1, 0, 0],
                [-0.5, np.sqrt(3) / 2, 0],
                [-0.5, -np.sqrt(3) / 2, 0]
            ])  # Already normalized, no need to divide further

        elif sp == 1:
            # sp is linear, bond angle 180°
            return np.array([
                [1, 0, 0],
                [-1, 0, 0]
            ])  # Already normalized

        return None

    def align_orbitals(self, atom:Atom, sp, preferred_atoms=[], strip_orbitals=False, try_align=True):
        """
        Computes a hybridization matrix for a two-shell atom.

        For core orbitals (like the 1s orbital), the corresponding rows/columns will be removed if `strip_orbitals` is True.

        For valence orbitals (in the 2s, 2px, 2py, and 2pz orbitals), the function will rotate hybrid orbitals
        to match bond directions as closely as possible.

        For single-shell atoms (like H), will return a 1x1 identity matrix.

        :param preferred_atoms: Atoms given priority in bond alignment (not used currently).
        :param strip_orbitals: Whether or not to remove core 1s orbitals (for simplification).
        """
        bonded_atoms = self.get_bonds(atom)
        if atom.number_of_shells == 1:  # For atoms that have only the 1s orbital, there is no hybridization.
            return np.eye(1)

        preferred_atoms_set = set(preferred_atoms)
        ordered_bonds = preferred_atoms[:]
        for bonded_atom in bonded_atoms:
            if bonded_atom not in preferred_atoms_set:
                ordered_bonds.append(bonded_atom)
        ###
        # Ignore ordered_bonds for now
        ###

        num_bonds = len(bonded_atoms)
        atom_coords = atom.coords
        vectors = np.array([b.coords - atom_coords for b in bonded_atoms])  # Calculating bond vectors

        # Handle case where there are no bonds.
        if num_bonds == 0:
            if atom.name == 'Hydrogen' or atom.name == 'Helium':
                return np.eye(1)
            return np.eye(5 if not strip_orbitals else 4)

        # Get the ideal hybrid orbital directions (e.g., sp, sp², sp³).
        hybrid_orbitals = self.hybridization_orbital_vectors(sp)

        if hybrid_orbitals is None or hybrid_orbitals.shape[0] == 0:
            raise ValueError(f"Invalid hybrid orbital configuration for sp type: {sp}")

        ### Begin constructing the hybridization matrix ###
        orbital_matrix = np.eye(5)  # Matrix size for two-shell atoms (1s, 2s, 2px, 2py, 2pz)

        # The 1s orbital stays unhybridized, so it remains untouched:
        # orbital_matrix[0, 0] = 1 (and the rest of the row/column is zero),
        # so no need to modify the first row or column.

        # Step 1: Combine s and p orbital contributions for the other orbitals.

        # We need to convert 3D vectors (representing bonds) to a 4-dimensional form.
        # The 4D form corresponds to: [s contribution, px contribution, py contribution, pz contribution].
        # For each bond direction vector, we'll create hybrid orbital components.

        num_hybrid_orbitals = len(hybrid_orbitals)

        # If there are more bond vectors than hybrid orbitals (highly unusual but just in case):
        effective_num_orbitals = num_bonds if num_bonds>num_hybrid_orbitals else num_hybrid_orbitals
        # Step 2: Create a hybrid orbital matrix with contributions from p orbitals.
        final_hybrid_orbitals = np.zeros((effective_num_orbitals, 3))  # 3D space for 2p interactions

        hybrid_weight = 1 / np.sqrt(sp + 1)  # 1/sqrt(sp + 1)
        p_weight = np.sqrt(sp / (sp + 1))  # sqrt(sp/(sp + 1))

        for idx, bond in enumerate(vectors[:effective_num_orbitals]):
            bond_norm = bond / np.linalg.norm(bond)  # Normalize bond vector
            final_hybrid_orbitals[idx] = [
                p_weight * bond_norm[0],  # px
                p_weight * bond_norm[1],  # py
                p_weight * bond_norm[2]  # pz
            ]

        # Align ideal hybrid orbital directions to bond vectors via Procrustes alignment.
        selected_hybrid_orbitals = hybrid_orbitals[:effective_num_orbitals]

        if (try_align): final_hybrid_orbitals = HybridizationUtils.kabsch_alignment(selected_hybrid_orbitals, final_hybrid_orbitals)
        # if (try_align):  final_hybrid_orbitals = HybridizationUtils.correct_signs(rotated_hybrids, final_hybrid_orbitals)
        placed = [False] + [True]*3  #1s always placed Modify if further orbitals
        # Replace the aligned results into the hybrid orbital matrix (s, px, py, pz contributions)
        for idx in range(effective_num_orbitals):
            if idx: #first always the s contribution
                nzero = np.argwhere(final_hybrid_orbitals[idx, :])
                for nz in nzero:
                    if placed[nz[0]]: #care 1s
                        placed[nz[0]]=False
                        orbital_matrix[2 + nz[0], 1] = hybrid_weight  # s contribution, 2 + nz bcs 1 (from 1s) + 1 (p )
                        orbital_matrix[2 + nz[0], 2:] = final_hybrid_orbitals[idx, :]  # rotated px, py, pz contributions
            else:
                orbital_matrix[1 + idx, 1] = hybrid_weight  # s contribution
                orbital_matrix[1, 2:] = final_hybrid_orbitals[0, :]  # rotated px, py, pz contributions
        if strip_orbitals:
            orbital_matrix = orbital_matrix[1:, 1:]

        # The orbital matrix now contains the combined s-p hybridization with orientation aligned to the bonds.
        return orbital_matrix

    def apply_hybridization(self, atom:Atom, sp='Auto', strip_orbitals=False):
        bonded_atoms = self.get_bonds(atom)
        if len(bonded_atoms) == 0 or sp == None:
            if atom.name == 'Hydrogen' or atom.name == 'Helium':
                return np.eye(1)
            return np.eye(5 if not strip_orbitals else 4)
        hit_data = None

        if sp == 'Auto':
            sp, all_hit_data = self.get_hybridization_data(atom)
            if all_hit_data is not None: hit_data = all_hit_data[sp]
        else:
            sp = int(sp)
            hit_data = self.get_hit_data(self.get_angle_data(atom), sp)

        if hit_data is None: hit_data = []

        if len(hit_data) == 0:
            anchor_atom = bonded_atoms[0]
        else:
            anchors = {}
            for angle, neighbor1, neighbor2 in hit_data:
                anchors[neighbor1] = anchors.get(neighbor1, 0) + 1
                anchors[neighbor2] = anchors.get(neighbor2, 0) + 1

            anchor_atom = max(anchors, key=anchors.get)

        # Align the bond with respect to the anchor_atom
        target_angles = {1: 180, 2: 120, 3: 109.5}
        bond_angles = [(angle, neighbor1, neighbor2) for (angle, neighbor1, neighbor2) in hit_data
                       if neighbor1 == anchor_atom or neighbor2 == anchor_atom]
        bond_angles.sort(key=lambda angle_data: abs(angle_data[0] - target_angles[sp]))

        # Collect preferred atoms
        preferred_atoms = [anchor_atom]
        for angle, neighbor1, neighbor2 in bond_angles:
            if neighbor1 != anchor_atom:
                preferred_atoms.append(neighbor1)
            if neighbor2 != anchor_atom:
                preferred_atoms.append(neighbor2)

        orbital_matrix = self.align_orbitals(atom, sp, preferred_atoms, strip_orbitals=strip_orbitals)
        return orbital_matrix

    def get_orbital_coefficient_matrix(self, strip_orbitals:bool=False) -> np.array:
        '''

        :param strip_orbitals: will skip 1s orbs
        :return:
        '''
        matrices = [self.apply_hybridization(atom, strip_orbitals=strip_orbitals) for atom in self.atoms]
        bond_data = [self.get_bonds(atom) for atom in self.atoms]
        size = sum(matrix.shape[0] for matrix in matrices)
        coefficient_matrix = np.zeros((size, size))
        # Fill the final matrix by placing each matrix along the diagonal
        current_index = 0
        matrix_indices = []
        for matrix in matrices:
            matrix_size = matrix.shape[0]  # Size of the individual matrix (5x5 or 1x1, etc.)
            coefficient_matrix[current_index:current_index + matrix_size,current_index:current_index + matrix_size] = matrix
            matrix_indices.append(current_index)
            current_index += matrix_size  # Move the starting point for the next block
        applied_bonds = set()  # To avoid double application of transformations
        rows_start = {i:matrix_indices[i] for i in range(len(self.atoms))}
        for i, atom_bonds in enumerate(bond_data):
            for j, neighbor_atom in enumerate(atom_bonds):
                neighbor_idx = self.atom_indices[neighbor_atom]
                # Only apply transformation once per bond
                bond_pair = tuple(sorted((i, neighbor_idx)))
                if bond_pair in applied_bonds:
                    continue
                applied_bonds.add(bond_pair)

                # neighbor_matrix_start = matrix_indices[neighbor_idx]

                # Determine which rows to modify (ignoring 1s orbital row in 5x5 matrices)
                atom_matrix_size = matrices[i].shape[0]
                neighbor_matrix_size = matrices[neighbor_idx].shape[0]

                if atom_matrix_size == 1:
                    bond_row_i = rows_start[i]  # Size 1, so only one orbital
                else:
                    bond_row_i = rows_start[i] + j + (not strip_orbitals) # Bonding row starts at 1 due to skipping 1s
                if neighbor_matrix_size == 1:
                    bond_row_neighbor = rows_start[neighbor_idx]  # Size 1, so only one orbital
                else:
                    bond_row_neighbor = rows_start[neighbor_idx] + bond_data[neighbor_idx].index(self.atoms[i]) + (not strip_orbitals) # Bonding row starts at 1 due to skipping 1s
                # Construct line addition transformation matrix for the bond
                transformation_matrix = np.eye(size)  # Start with identity matrix
                transformation_matrix[bond_row_i, bond_row_neighbor] = 1  # 1 for bond alignment
                transformation_matrix[bond_row_neighbor, bond_row_i] = -1  # Symmetry
                # theta = np.pi / 2  # 30 degree rotation (can be adjusted)
                # transformation_matrix = MatrixUtils.givens_rotation_matrix(size, bond_row_i, bond_row_neighbor, theta)

                # Apply this transformation to the overall matrix by multiplying
                coefficient_matrix = np.dot(transformation_matrix, coefficient_matrix)
        del current_index
        positions = np.array([atom.coords for atom in self.atoms])
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        del positions
        distances = np.sqrt(np.sum(diff ** 2, axis=2))
        del diff
        indices = {i:len(bond_data[i]) for i in range(len(self.atoms)) if len(bond_data[i])<self.atoms[i].max_bonds}  
        for i in indices:
            # a dictionary is created with {neigh:dist_to_neigh} for all atoms bonded to the current one. 
            # We only consider them if not considered already (first condicional), 
            # there are still more atoms to assign (second conditional) 
            # and the neigh-atom is more than single bonded, since it is already considered above (third conditional) 
            bonded = {self.atom_indices[neighbor_atom]:distances[i,self.atom_indices[neighbor_atom]] for neighbor_atom in bond_data[i] if self.atom_indices[neighbor_atom]> i and len(bond_data[self.atom_indices[neighbor_atom]]) < neighbor_atom.max_bonds and self.get_bond_multiplicity(self.atoms[i],neighbor_atom)>1}
            bonded = dict(sorted(bonded.items(), key=lambda item: item[1]))
            for j in bonded:
                neighbor_atom = self.atoms[j]
                multi = self.get_bond_multiplicity(self.atoms[i],neighbor_atom)
                b = 0
                while b < multi-1:
                    bond_row_i = rows_start[i] + len(self.get_bonds(self.atoms[i])) + (not strip_orbitals) + b
                    bond_row_neighbor = rows_start[j] + len(self.get_bonds(self.atoms[i])) + (not strip_orbitals) + b
                    transformation_matrix = np.eye(size)  # Start with identity matrix
                    transformation_matrix[bond_row_i, bond_row_neighbor] = 1  # 1 for bond alignment
                    transformation_matrix[bond_row_neighbor, bond_row_i] = -1  # Symmetry
                    indices[j] += 1
                    indices[i] += 1
                    b +=1
                    coefficient_matrix = np.dot(transformation_matrix, coefficient_matrix)
        return coefficient_matrix

    def get_HAO_orbitals(self) -> np.array:
        matrices = [self.apply_hybridization(atom, strip_orbitals=False) for atom in self.atoms]
        size = sum(matrix.shape[0] for matrix in matrices)
        coefficient_matrix = np.zeros((size, size))
        # Fill the final matrix by placing each matrix along the diagonal
        current_index = 0
        matrix_indices = []
        for matrix in matrices:
            matrix_size = matrix.shape[0]  # Size of the individual matrix (5x5 or 1x1, etc.)
            coefficient_matrix[current_index:current_index + matrix_size,current_index:current_index + matrix_size] = matrix
            matrix_indices.append(current_index)
            current_index += matrix_size  # Move the starting point for the next block
        return coefficient_matrix

    def get_spa_edges(self, collapse=True,strip_orbitals:bool=True):
        """
        Generates SPA edges for a molecule, handling per-shell electrons assignment,
        bonding prioritization, lone pair behavior, and shell-based electron distribution.

        :param collapse: If True, lone pairs collapse into a single index per pair,
                        otherwise each electron retains its own index.
        :param strip_orbitals: If True, fully occupied orbitals (Frozen core) are omited
        :return: List of edges, where each entry represents either a bonding connection or a lone pair.
        """
        index = 0  # Global index for assigning electrons
        edges = []  # List to hold the final SPA edges
        edges_to_append = {}  # Tracks remaining electrons for bonds that haven't been finalized yet
        for atom in self.atoms:
            bonds = self.get_multiplied_bonds(atom)  # Get bonded atoms (bonding partners)
            remaining_bonds = len(bonds)
            bond_index = 0
            electrons_by_shell = atom.electrons_by_shell  # Electron configuration per shell
            max_electrons_by_shell = AtomUtils.max_electrons_by_shell  # Shell electron limits
            if strip_orbitals:
                ebs = deepcopy(electrons_by_shell)
                mebs = deepcopy(max_electrons_by_shell)
                for i in range(atom.number_of_shells):
                    if electrons_by_shell[i] == max_electrons_by_shell[i]: #Shell i fully occupied
                        ebs.pop(i)
                        mebs.pop(i)
                electrons_by_shell = ebs
                max_electrons_by_shell = mebs
            for shell_idx, electrons_in_shell in enumerate(electrons_by_shell):
                if shell_idx >= len(max_electrons_by_shell):
                    break
                max_bonding_capacity = max_electrons_by_shell[shell_idx] // 2 #max_bond = when shell half filled = number orbs
                bonding_capacity = max_bonding_capacity
                if electrons_in_shell > max_bonding_capacity: # if more than half filled WHY ONLY IF BIGGER, NOT SMALLER
                    bonding_capacity -= electrons_in_shell - max_bonding_capacity # -> bonding_cap = num unpaired electrons
                remaining_bonding_capacity = bonding_capacity
                remaining_unbondable_electrons = electrons_in_shell - bonding_capacity #unbondable e- = number paired e- (can be neg -> unpaired e-)
                # Step 1: Assign bondable electrons to bonds -> {atom_i:[b_j,b_k,...]}
                while (remaining_bonding_capacity > 0 and remaining_bonds > 0):
                    bonded_atom = bonds[bond_index] #current bonding partner
                    edges_to_append.setdefault(atom, {}) # placeholder, bond with only one side -> {a_i:[]}
                    edges_to_append[atom].setdefault(bonded_atom,[]) # {a_i,[b_j,b_k,...]}
                    edges_to_append[atom][bonded_atom].append(index)
                    remaining_bonding_capacity -= 1
                    remaining_bonds -= 1
                    bond_index += 1 #next bond
                    index += 1 #next electron
                remaining_unbondable_electrons += remaining_bonding_capacity
                # Step 2: Assign remaining electrons as lone pairs, core orbitals or unbonded
                while remaining_unbondable_electrons > 0:
                    if collapse and remaining_unbondable_electrons >= 2:
                        # Collapse two electrons into a single lone pair index
                        edges.append((index,))
                        remaining_unbondable_electrons -= 2
                        index += 1
                    else:
                        # Assign a single electron
                        edges.append((index,))
                        remaining_unbondable_electrons -= 1
                        index += 1
        for atom in edges_to_append:
            for bond in edges_to_append[atom]:
                assert bond in edges_to_append[atom]
                assert atom in edges_to_append[bond]
                assert len(edges_to_append[atom][bond])==len(edges_to_append[bond][atom])
                for i in range(len(edges_to_append[atom][bond])):
                    edges.append(tuple(sorted((edges_to_append[atom][bond][i],edges_to_append[bond][atom][i]))))
                edges_to_append[atom][bond]=[]
                edges_to_append[bond][atom]=[]
        return edges


    def rotate_around_point(self, angle, axis, point=None):
        """
        Rotates the entire molecule around a given axis and an optional reference point.

        :param angle: Rotation angle in degrees.
        :param axis: A 3D vector (array or list) denoting the axis of rotation.
                     Example: [1, 0, 0] for the x-axis, [0, 1, 0] for the y-axis, etc.
        :param point: A reference point for rotation in space (array or list). Defaults to None, which sets it to zero.
        """
        # Convert angle to radians
        angle_rad = np.radians(angle)

        # Normalize the axis vector
        axis = axis / np.linalg.norm(axis)

        # If no point is given, set to origin
        if point is None:
            point = np.array([0.0, 0.0, 0.0])
        else:
            point = np.array(point)

        # Rotation matrix using Rodrigues' rotation formula
        cos_theta = np.cos(angle_rad)
        sin_theta = np.sin(angle_rad)
        ux, uy, uz = axis

        # Constructing the rotation matrix
        rotation_matrix = np.array([
            [cos_theta + ux ** 2 * (1 - cos_theta), ux * uy * (1 - cos_theta) - uz * sin_theta,
             ux * uz * (1 - cos_theta) + uy * sin_theta],
            [uy * ux * (1 - cos_theta) + uz * sin_theta, cos_theta + uy ** 2 * (1 - cos_theta),
             uy * uz * (1 - cos_theta) - ux * sin_theta],
            [uz * ux * (1 - cos_theta) - uy * sin_theta, uz * uy * (1 - cos_theta) + ux * sin_theta,
             cos_theta + uz ** 2 * (1 - cos_theta)]
        ])

        # Shift positions to align with the rotation point
        shifted_positions = np.array([atom.coords - point for atom in self.atoms])

        # Apply rotation matrix to the shifted positions
        rotated_positions = np.dot(rotation_matrix, shifted_positions.T).T

        # Shift back to original coordinates
        final_positions = rotated_positions + point

        # Update atom positions with the new rotated positions
        for atom, new_coords in zip(self.atoms, final_positions):
            atom.position.x, atom.position.y, atom.position.z = new_coords

    def get_geometry_string(self):
        return '\n'.join([atom.to_string() for atom in self.atoms])

    def to_xyz(self):
        return f"{len(self.atoms)}\n" + "Parsed Molecule\n" + self.get_geometry_string()

    def debug(self):
        for atom in self.atoms:
            bonded_atoms = self.get_bonds(atom)
            bond_parts = []
            for bond in bonded_atoms:
                multiplicity = self.get_bond_multiplicity(atom, bond)
                part = bond.to_string()
                if multiplicity != 1: part += f" ({multiplicity})"
                bond_parts.append(part)
            bond_str = ', '.join(bond_parts)

            hybridization = self.get_hybridization(atom)
            hybridization_str = f"sp{hybridization}" if hybridization else "None"
            orbital_matrix = self.apply_hybridization(atom, hybridization)

            print(f"{atom.to_string()} is bonded to: {bond_str}")
            print(f"Hybridization: {hybridization_str}\n")
            print(f"With Orbital Matrix: {orbital_matrix}\n")
        coefficient_matrix = self.get_orbital_coefficient_matrix()
        print(f"Full Coefficient Matrix: {coefficient_matrix}\n")
