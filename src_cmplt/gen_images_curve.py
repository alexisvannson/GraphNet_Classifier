import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

def generate_2Dcurve_image(a, b, c, idx):
    x = np.linspace(-10, 10, 100)
    y = a * x**2 + b * x + c
    
    plt.figure(figsize=(2, 2), dpi=32)
    plt.plot(x, y, color='black', linewidth=3)
    plt.ylim(-200, 200)
    plt.axis('off')
    plt.tight_layout()
    path = f"images/img_{idx:03}.png"
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()
    return path


def generate_3dim_image(a, b, c, ax, idx, rotation):
    """Generate 3D plot on given axis instead of creating new figure"""
    x = np.linspace(-6, 6, 30)
    y = np.linspace(-6, 6, 30)
    X, Y = np.meshgrid(x, y)
    Z = a * (X + Y)**2 + b * (X + Y) + c
    
    tri = Triangulation(X.ravel(), Y.ravel())
    
    ax.plot_trisurf(tri, Z.ravel(), cmap='cool', edgecolor='none', alpha=0.8)
    ax.grid(False)
    ax.xaxis.pane.set_visible(False)
    ax.yaxis.pane.set_visible(False)
    ax.zaxis.pane.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    match rotation:
        case 'XY':
            ax.set_title('Surface Plot from XY')
            ax.view_init(elev=90, azim=-90)
        case 'XZ':
            ax.set_title('Surface Plot from XZ')
            ax.view_init(elev=0, azim=0)
        case 'YZ':
            ax.set_title('Surface Plot from YZ')
            ax.view_init(elev=0, azim=-90)

def create_combined_3d_plot(a, b, c, idx):
    """Create combined 3D plot and save it"""
    # Create the subplot figure with 3 3D axes
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': '3d'})
    
    # Plot the three rotations on the respective axes
    generate_3dim_image(a, b, c, axes[0], idx, 'XY')
    generate_3dim_image(a, b, c, axes[1], idx, 'XZ')
    generate_3dim_image(a, b, c, axes[2], idx, 'YZ')
    
    plt.tight_layout()
    path = f"images3dim/img_{idx:03}.png"
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()
    return path