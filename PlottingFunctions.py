# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

MEDIUM_FONT = 14
LARGE_FONT = MEDIUM_FONT + 2
SMALL_FONT = MEDIUM_FONT + 2
figsize = None #(12,9)
figsize_square = None #(9,9)


def plot_field(field, cube, z, orientation, fig = None, ax = None):
    """"
    input: z = float between 0,1
            orientation = [0,1,2] corresponding with the x,y,z direction
    """
    label = ['x', 'y', 'z']

    if (not fig) and (not ax):
        fig, ax = plt.subplots(1,1, figsize=figsize)

    z_index = int(np.floor(z*cube.resolution))

    ax.set_title(f'field in the {label[orientation]}-direction', fontsize = LARGE_FONT)
    ax.set_xlabel('x [m]', fontsize=MEDIUM_FONT)
    ax.set_ylabel('y [m]', fontsize=MEDIUM_FONT)
    ax.tick_params(axis='both', which='major', labelsize=MEDIUM_FONT)
    ax.tick_params(axis='both', which='minor', labelsize=SMALL_FONT)

    pc = ax.pcolormesh(cube.x_arr, cube.y_arr, field[:,:,z_index,orientation])
    fig.colorbar(pc)
    return fig, ax


def plot_field_arrow(field, cube, frac, orientation, fig = None, ax = None):
    """"
    input: z = float between 0,1
            orientation = [0,1,2] corresponding with viewing the xy, yz and xz plane
    """
    plane_label = ['xy', 'yz', 'xz']
    slice_label = ['z', 'x', 'y']
    x_label = ['x [m]', 'y [m]', 'x [m]']
    y_label = ['y [m]', 'z [m]', 'z [m]']

    if (not fig) and (not ax):
        fig, ax = plt.subplots(1,1, figsize=figsize)

    frac_index = int(np.floor(frac*cube.resolution))

    ax.set_xlabel(x_label[orientation], fontsize=MEDIUM_FONT)
    ax.set_ylabel(y_label[orientation], fontsize=MEDIUM_FONT)
    ax.tick_params(axis='both', which='major', labelsize=MEDIUM_FONT)
    ax.tick_params(axis='both', which='minor', labelsize=SMALL_FONT)

    scale = np.max(field)
    U_x = 0.25*field[:,:,:,0]/scale
    U_y = 0.25*field[:,:,:,1]/scale
    U_z = 0.25*field[:,:,:,2]/scale
    
    if orientation == 0:
        ax.quiver(cube.X[:,:,frac_index], cube.Y[:,:,frac_index], U_x[:,:,frac_index], U_y[:,:,frac_index])
        pos = cube.Z[0,0,frac_index]
    elif orientation == 1:
        ax.quiver(cube.Y[:,frac_index,:], cube.Z[:,frac_index,:], U_y[:,frac_index,:], U_z[:,frac_index,:])
        pos = cube.X[0,frac_index,0]
    else:
        ax.quiver(cube.X[frac_index,:,:], cube.Z[frac_index,:,:], U_x[frac_index,:,:], U_z[frac_index,:,:])
        pos = cube.Y[frac_index,0,0]

    ax.set_title(f'field in the {plane_label[orientation]}-direction at {slice_label[orientation]} = {pos:.2f} m', fontsize = LARGE_FONT)
    
    return fig, ax


def plot_field_arrow_3d(field, cube, fig = None, ax = None):
    """"
    input:
    """
    if (not fig) and (not ax):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection='3d')

    ax.set_xlabel('x [m]', fontsize=MEDIUM_FONT)
    ax.set_ylabel('y [m]', fontsize=MEDIUM_FONT)
    ax.set_zlabel('z [m]', fontsize=MEDIUM_FONT)
    ax.tick_params(axis='both', which='major', labelsize=MEDIUM_FONT)
    ax.tick_params(axis='both', which='minor', labelsize=SMALL_FONT)
    ax.yaxis._axinfo['label']['space_factor'] = 10.0
    ax.xaxis._axinfo['label']['space_factor'] = 10.0
    ax.set_box_aspect(None, zoom=0.85)

    plt.margins(0.1)
    # scale = np.max(np.abs(self.field))
    scale = np.max(field)
    
    U_x = 0.25*field[:,:,:,0]/scale
    U_y = 0.25*field[:,:,:,1]/scale
    U_z = 0.25*field[:,:,:,2]/scale

    ax.quiver(cube.X, cube.Y, cube.Z, U_x, U_y, U_z)

    return ax


def plot_contour(pcb, fig = None, ax = None):
    if (not fig) and (not ax):
        fig, ax = plt.subplots(1,1, figsize=figsize)

    ax.set_xlabel('x [m]', fontsize=MEDIUM_FONT)
    ax.set_ylabel('y [m]', fontsize=MEDIUM_FONT)

    for c in pcb.contours:
        c.plot_contour(fig, ax)
