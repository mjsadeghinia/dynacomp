import gmsh
import sys
import os

def main():
    # Initialize Gmsh
    gmsh.initialize()
    
    # Optionally output messages to the terminal
    gmsh.option.setNumber("General.Terminal", 1)
    
    # Create a new Gmsh model
    gmsh.model.add("SurfaceMesh")

    # Path to your PLY file
    stl_file = "endo.stl"  # Replace with your PLY file name

    # Check if the file exists
    if not os.path.isfile(stl_file):
        print(f"Error: File '{stl_file}' not found.")
        gmsh.finalize()
        return

    # Import the PLY file
    gmsh.merge(stl_file)

    # Classify surfaces to prepare for meshing
    # Here, we define the angle (in degrees) between two triangles that will be considered sharp
    angle = 40

    # Force the mesh elements to be classified on discrete entities
    # (surfaces, curves) that respect the sharp edges
    gmsh.model.mesh.classifySurfaces(angle * (3.141592653589793 / 180.0), True, False, 0.01, True)

    # Create geometry from the classified mesh
    gmsh.model.mesh.createGeometry()

    # Create a volume from surfaces (if needed)
    # gmsh.model.geo.addVolume(...)

    # Synchronize the built-in CAD kernel with the Gmsh model
    gmsh.model.geo.synchronize()

    # Set mesh size (optional)
    # gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 10)
    # gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 10)
    
    
    # Option 3: Use a distance-based mesh size field (advanced control)
    # Uncomment and adjust as needed
    points_of_interest = gmsh.model.getEntities(0)  # Get all points
    distance_field = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance_field, "NodesList", [p[1] for p in points_of_interest])
    threshold_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold_field, "InField", distance_field)
    gmsh.model.mesh.field.setNumber(threshold_field, "SizeMin", 1)
    gmsh.model.mesh.field.setNumber(threshold_field, "SizeMax", 10)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", 0)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 10)
    gmsh.model.mesh.field.setAsBackgroundMesh(threshold_field)

    # Generate 2D mesh
    gmsh.model.mesh.generate(2)

    # Optionally, generate 3D mesh (if you need volume elements)
    # gmsh.model.mesh.generate(3)

    # Save the mesh to a file
    output_file = "gmsh_output.stl"
    gmsh.write(output_file)
    print(f"Mesh saved to '{output_file}'")

    # Finalize Gmsh
    gmsh.finalize()

if __name__ == "__main__":
    main()
