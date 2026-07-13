import numpy as np

from gaugeflow.unit_cell import niggli_reduce_structure
from pymatgen.core import Lattice, Structure


def test_niggli_reduction_tracks_fractional_coordinates_with_unimodular_basis():
    lattice = Lattice([[3.0, 0.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 5.0]])
    structure = Structure(lattice, ["Si", "O"], [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    reduced = niggli_reduce_structure(structure)
    change = reduced.lattice.matrix @ np.linalg.inv(structure.lattice.matrix)
    assert np.allclose(change, np.rint(change), atol=1e-5)
    assert abs(round(np.linalg.det(change))) == 1
    displacement = reduced.frac_coords @ reduced.lattice.matrix - structure.frac_coords @ structure.lattice.matrix
    lattice_translation = displacement @ np.linalg.inv(structure.lattice.matrix)
    assert np.allclose(lattice_translation, np.rint(lattice_translation), atol=1e-5)
