import math
import numpy as np
from diffusion import calculate_time_step, set_initial_profile, make_grid

GRID_SPACING = 1.0
DIFFUSIVITY = 1.0
TIME_STEP = 0.5
TOLERANCE = 0.01
GRID_SIZE = 100
BNDY_LEFT = 500
BNDY_RIGHT = 0
ORIGIN = 0
DOMAIN_SIZE = GRID_SIZE * GRID_SPACING



def test_time_step():
    time_step = calculate_time_step(GRID_SPACING, DIFFUSIVITY)
    assert type(time_step) is float
    assert math.isclose(time_step, TIME_STEP, rel_tol=TOLERANCE)


def test_initial_profile():
    profile = set_initial_profile()
    assert type(profile) is np.ndarray
    assert len(profile) == GRID_SIZE
    assert math.isclose(profile[0], BNDY_LEFT)
    assert math.isclose(profile[-1], BNDY_RIGHT)


def test_grid_length():
    grid, grid_length = make_grid(ORIGIN, DOMAIN_SIZE, GRID_SPACING)
    assert type(grid) is np.array
    assert type(grid_length) is int

