"""
Standalone utility functions extracted from the ssneb class.

These functions implement the core geometry operations used in solid-state NEB:
linear interpolation, Jacobian scaling, image property initialization, and the
inter-image distance metric. They operate on ASE Atoms objects and numpy arrays
without requiring an ssneb instance.
"""

import numpy
from tsase.neb.util import sPBC


def compute_jacobian(vol1, vol2, natom, weight=1.0):
    """Compute the Jacobian scaling factor for solid-state NEB.

    The Jacobian balances the units and weight of cell deformations against
    atomic displacements, so that the NEB spring forces treat both on equal
    footing.

    Args:
        vol1: Volume of the first endpoint cell.
        vol2: Volume of the second endpoint cell.
        natom: Number of atoms per image.
        weight: Relative weight of cell vs atomic degrees of freedom.

    Returns:
        The Jacobian scaling factor (float).
    """
    vol = (vol1 + vol2) * 0.5
    avglen = (vol / natom) ** (1.0 / 3.0)
    return avglen * natom ** 0.5 * weight


def interpolate_path(p1, p2, num_images):
    """Linearly interpolate a path between two endpoint structures.

    Interpolates both the cell vectors and fractional atomic coordinates.
    Fractional coordinate differences are wrapped via sPBC so that atoms
    take the shortest periodic path.

    Args:
        p1: ASE Atoms object for the first endpoint.
        p2: ASE Atoms object for the second endpoint.
        num_images: Total number of images including both endpoints.

    Returns:
        List of num_images ASE Atoms objects. The first and last are copies
        of p1 and p2 respectively; intermediates are interpolated.
    """
    n = num_images - 1
    cell1 = p1.get_cell()
    cell2 = p2.get_cell()
    dRB = (cell2 - cell1) / n

    # Use Cartesian -> fractional via inverse cell (not get_scaled_positions)
    # so that atoms moving more than half a lattice vector are handled correctly.
    icell1 = numpy.linalg.inv(cell1)
    vdir1 = numpy.dot(p1.get_positions(), icell1)
    icell2 = numpy.linalg.inv(cell2)
    vdir2 = numpy.dot(p2.get_positions(), icell2)
    dR = sPBC(vdir2 - vdir1) / n

    path = [p1.copy()]
    for i in range(1, n):
        img = p1.copy()
        cellt = cell1 + dRB * i
        vdirt = vdir1 + dR * i
        rt = numpy.dot(vdirt, cellt)
        img.set_cell(cellt)
        img.set_positions(rt)
        path.append(img)
    path.append(p2.copy())
    return path


def initialize_image_properties(image, jacobian):
    """Attach derived NEB properties to an ASE Atoms image.

    Sets three attributes used throughout the ssneb force calculation:
      - image.cellt: cell matrix scaled by the Jacobian
      - image.icell: inverse of the (unscaled) cell matrix
      - image.vdir:  fractional (scaled) positions

    Args:
        image: An ASE Atoms object.
        jacobian: The Jacobian scaling factor from compute_jacobian().
    """
    image.cellt = numpy.array(image.get_cell()) * jacobian
    image.icell = numpy.linalg.inv(image.get_cell())
    image.vdir = image.get_scaled_positions()


def image_distance_vector(image_a, image_b):
    """Compute the combined atom + cell distance vector between two images.

    This is the metric used by ssneb for spring forces. It combines:
      - Atomic part: sPBC-wrapped fractional coordinate differences converted
        to Cartesian via the average cell matrix. Shape (natom, 3).
      - Cell part: Jacobian-scaled cell differences averaged through both
        images' inverse cells. Shape (3, 3).

    The two parts are vertically stacked into a single array whose Euclidean
    norm (via vmag) gives the scalar inter-image distance.

    Both images must have .cellt, .icell, and .vdir attributes set (see
    initialize_image_properties).

    Args:
        image_a: ASE Atoms with NEB properties (the "from" image).
        image_b: ASE Atoms with NEB properties (the "to" image).

    Returns:
        numpy array of shape (natom + 3, 3) — the stacked distance vector.
    """
    # Atomic part: fractional diff -> Cartesian via average cell
    frac_diff = sPBC(image_a.vdir - image_b.vdir)
    avgbox = 0.5 * (numpy.array(image_a.get_cell()) + numpy.array(image_b.get_cell()))
    atom_part = numpy.dot(frac_diff, avgbox)

    # Cell part: Jacobian-scaled cell diff averaged through inverse cells
    dh = image_a.cellt - image_b.cellt
    cell_part = numpy.dot(image_b.icell, dh) * 0.5 + numpy.dot(image_a.icell, dh) * 0.5

    return numpy.vstack((atom_part, cell_part))
