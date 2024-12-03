import numpy as np
import open3d as o3d
import os

def load_point_cloud(file_path):
    """
    Load point cloud data from a CSV file, skipping the first row.
    Each row after the first should contain x, y, z coordinates.
    """
    # Load the CSV file, skipping the first row
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)

    # Check if data has at least three columns
    if data.shape[1] < 3:
        raise ValueError("CSV file must contain at least three columns for x, y, z coordinates.")

    # Extract x, y, z coordinates
    points = data[:, :3]

    # Create an Open3D point cloud from the numpy array
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    return pcd

def preprocess_point_cloud(pcd):
    """
    Preprocess the point cloud: estimate normals and orient them consistently.
    """
    # Estimate normals
    pcd.estimate_normals()

    # Orient normals consistently
    pcd.orient_normals_consistent_tangent_plane(k=10)

    return pcd

def create_surface_mesh(pcd):
    """
    Create a surface mesh from the point cloud using Poisson reconstruction.
    """
    # Perform Poisson surface reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=5
    )

    # Optional: Clean up the mesh
    mesh.remove_non_manifold_edges()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()

    return mesh

def main():
    # Load point cloud data from CSV file
    file_path = 'endo.csv'  # Ensure this is the correct path to your CSV file
    pcd = load_point_cloud(file_path)

    # Preprocess the point cloud
    pcd = preprocess_point_cloud(pcd)

    # Create surface mesh from point cloud
    mesh = create_surface_mesh(pcd)

    # Compute normals for the mesh (required for STL export)
    mesh.compute_vertex_normals()

    # Visualize the mesh
    # o3d.visualization.draw_geometries([mesh])

    # Save the mesh as STL
    base_name = os.path.splitext(file_path)[0]
    fname = base_name + '.stl'
    o3d.io.write_triangle_mesh(fname, mesh)
    print(f"Mesh saved as {fname}")

if __name__ == '__main__':
    main()
