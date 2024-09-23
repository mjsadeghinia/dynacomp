import numpy as np
import open3d as o3d
import gmsh
from pathlib import Path


def make_open3d_point_cloud(points: np.array):
    # Create an Open3D point cloud from the numpy array
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    return pcd

def preprocess_point_cloud(pcd, k=10):
    """
    Preprocess the point cloud: estimate normals and orient them consistently.
    """
    # Estimate normals
    pcd.estimate_normals()

    # Orient normals consistently
    pcd.orient_normals_consistent_tangent_plane(k=k)

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

def create_and_export_stl_surface(points, stl_fname):
    
    pcd = make_open3d_point_cloud(points)
    # Preprocess the point cloud
    pcd = preprocess_point_cloud(pcd)
    # Create surface mesh from point cloud
    mesh = create_surface_mesh(pcd)
    # Compute normals for the mesh (required for STL export)
    mesh.compute_vertex_normals()
    # Save the mesh as STL
    o3d.io.write_triangle_mesh(stl_fname, mesh)

        
def optimize_stl_mesh(stl_fname):
    gmsh.initialize()
    
    # Optionally output messages to the terminal
    gmsh.option.setNumber("General.Terminal", 1)
    
    # Create a new Gmsh model
    gmsh.model.add("SurfaceMesh")


    # Check if the file exists
    stl_path = Path(stl_fname)
    if not stl_path.is_file():
        print(f"Error: File '{stl_fname}' not found.")
        gmsh.finalize()
        return

    # Import the PLY file
    gmsh.merge(stl_fname)

    # Classify surfaces to prepare for meshing
    # Here, we define the angle (in degrees) between two triangles that will be considered sharp
    angle = 40

    # Force the mesh elements to be classified on discrete entities
    # (surfaces, curves) that respect the sharp edges
    gmsh.model.mesh.classifySurfaces(angle * (3.141592653589793 / 180.0), True, False, 0.01, True)

    # Create geometry from the classified mesh
    gmsh.model.mesh.createGeometry()


    # Synchronize the built-in CAD kernel with the Gmsh model
    gmsh.model.geo.synchronize()

    # Set mesh size (optional)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 10)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 10)
    
    
    # Alternative Option: Use a distance-based mesh size field (advanced control)
    # Uncomment and adjust as needed
    # points_of_interest = gmsh.model.getEntities(0)  # Get all points
    # distance_field = gmsh.model.mesh.field.add("Distance")
    # gmsh.model.mesh.field.setNumbers(distance_field, "NodesList", [p[1] for p in points_of_interest])
    # threshold_field = gmsh.model.mesh.field.add("Threshold")
    # gmsh.model.mesh.field.setNumber(threshold_field, "InField", distance_field)
    # gmsh.model.mesh.field.setNumber(threshold_field, "SizeMin", 1)
    # gmsh.model.mesh.field.setNumber(threshold_field, "SizeMax", 10)
    # gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", 0)
    # gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 10)
    # gmsh.model.mesh.field.setAsBackgroundMesh(threshold_field)

    # Generate 2D mesh
    gmsh.model.mesh.generate(2)

    # Optionally, generate 3D mesh (if you need volume elements)
    # gmsh.model.mesh.generate(3)

    # Save the mesh to a file
    gmsh.write(stl_fname)

    # Finalize Gmsh
    gmsh.finalize()