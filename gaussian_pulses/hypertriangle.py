"""
Copied from balrog for now
"""

import numpy as np

def hypertriangulate(x, bounds=(0, 1)):
    """
    Transform a vector of numbers from the hypercube to the hypertriangle.

    The hypercube is the space the samplers usually work in; the 
    components of x are in no particular order.
    
    The hypertriangle is the space where the components are sorted into
    ascenting order, y0 < y1 < ... < yn. 
    
    The (unit) transformation is defined by:

    .. math::
        y_j = 1 - \\prod_{i=0}^{j} (1 - x_i)^{1/(n-i)}

    Example application. If we are analysing a number num_dim of DWD 
    sources, all with identical priors. Then this function would be
    called on the array np.array([f_1, f_2, ..., f_num_sources]) with
    bounds=(f_min, f_max).

    Parameters
    ----------
    x: array
        The hypercube parameter values. 
        The components of x are in no particular order.
        Input shape = (num_dim,) or (num_points, num_dim).
        If input array is multi-dimensional, the function is vectorised 
        along all except the last axis.

    bounds: tuple
        Lower and upper bounds of parameter space. Default is to transform
        between the unit hypercube and unit hypertriangle with (0, 1).

    Returns
    -------
    y: array, shaped like x
        The hypertriangle parameter values
        The components of y are sorted in ascending order.
    """
    x = np.asarray(x)
    
    # transform to the unit hypercube
    unit_x = (x - bounds[0]) / (bounds[1] - bounds[0])

    # hypertriangle transformation
    n = np.size(unit_x, axis=-1)
    index = np.arange(n)
    inner_term = np.power(1 - unit_x, 1/(n - index))
    unit_y = 1 - np.cumprod(inner_term, axis=-1)

    # re-apply orginal scaling, offset
    y = bounds[0] + unit_y * (bounds[1] - bounds[0])

    return y


def hypercubify(y, bounds=(0, 1)):
    """
    Inverse of hypertriangulate. Maps from the hypertriangle to the hypercube.
    
    The (unit) transformation is defined by:

    .. math::
        x_i = 1 - [(1 - y_i)/(1 - y_{i-1})]^{n-i}

    Parameters
    ----------
    y: array
        The hypertriangle parameter values.
        The components of y are sorted in ascending order.
        Input shape = (num_dim,) or (num_points, num_dim).
        If input array is multi-dimensional, the function is vectorised 
        along all except the last axis.

    bounds: tuple
        Lower and upper bounds of parameter space. Default is to transform
        between the unit hypercube and unit hypertriangle with (0, 1).

    Returns
    -------
    x: array, shaped like x
        The hypercube parameter values. 
        The components of x are in no particular order.
    """
    y = np.asarray(y)

    # transform to the unit hypertriangle
    unit_y = (y - bounds[0]) / (bounds[1] - bounds[0])

    # inverse hypertriangle transformation
    n = np.size(unit_y, axis=-1)
    index = np.arange(n)
    unit_y_shifted = np.roll(unit_y, 1, axis=-1)
    unit_y_shifted[...,0] = 0
    unit_x = 1 - np.power((1 - unit_y) / (1 - unit_y_shifted), n - index)

    # re-apply orginal scaling, offset
    x = bounds[0] + unit_x * (bounds[1] - bounds[0])

    return x
