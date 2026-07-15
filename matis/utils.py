from collections.abc import Sequence
from dolfinx.fem.forms import Form
from dolfinx.fem.forms import extract_function_spaces as _extract_function_spaces
from dolfinx.fem.function import Function as _Function
import functools
import dolfinx.cpp as _cpp

from petsc4py import PETSc

import numpy as np
from numpy import typing as npt


def _dm_create_matrix(
    J: PETSc.Mat,  # type: ignore[name-defined]
    _dm: PETSc.DM,  # type: ignore[name-defined]
):
    """Return a clone of the matrix.

    Args:
        _dm: The DM instance.
        J: Matrix to assemble the Jacobian into.

    Returns:
        A PETSc matrix.
    """

    return J.duplicate()


def _dm_create_field_decomposition(
    u: npt.Union[Sequence[_Function], _Function],
    form: npt.Union[Form, Sequence[Form]],
    _dm: PETSc.DM,  # type: ignore[name-defined]
):
    """Return index sets of the fields and their associated names.

    Args:
        u: Function tied to the solution vector.
        form: Form of the residual or of the right-hand side.
            It can be a sequence of forms.
        _dm: The DM instance.

    Returns:
        names: field names.
        ises: list of index sets in global numbering.
        dms: list of subDMs. This function returns `None`.
    """

    if not isinstance(form, Sequence):
        form = [form]
    spaces = _extract_function_spaces(form)
    ises = _cpp.la.petsc.create_global_index_sets(
        [(V.dofmaps[0].index_map, V.dofmaps[0].index_map_bs) for V in spaces]  # type: ignore[union-attr]
    )
    if isinstance(u, Sequence):
        names = [f"{v.name + '_' if v.name != 'f' else ''}{i}" for i, v in enumerate(u)]
    else:
        names = [f"dolfinx_field_{i}" for i in range(len(form))]
    return names, ises, None
