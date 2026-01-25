import trimesh
import numpy as np
import argparse
import os

def voxelize(mesh_path, pitch=0.02, output_path=None):
    print(f"Loading mesh from {mesh_path}...")
    mesh = trimesh.load(mesh_path)
    
    # Ensure it's a single mesh
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    
    # Check if mesh has vertex colors
    has_colors = mesh.visual.kind == 'vertex' and mesh.visual.vertex_colors is not None
    print(f"Mesh has vertex colors: {has_colors}")
        
    print(f"Voxelizing with pitch={pitch}...")
    # Voxelize the mesh
    voxels = mesh.voxelized(pitch=pitch)
    
    # Fill the internal part if needed
    voxels = voxels.fill()
    
    print(f"Voxel grid shape: {voxels.shape}")
    print(f"Number of filled voxels: {voxels.matrix.sum()}")
    
    # Get voxel centers
    voxel_centers = voxels.points
    print(f"Number of voxel centers: {len(voxel_centers)}")
    
    # Create colored voxel mesh
    if output_path:
        print(f"Creating voxel mesh with colors...")
        
        if has_colors:
            # Sample colors from original mesh at voxel centers
            # Find closest point on mesh surface for each voxel center
            closest_points, distances, face_ids = mesh.nearest.on_surface(voxel_centers)
            
            # Get vertex colors for faces
            face_vertices = mesh.faces[face_ids]  # Shape: (n_voxels, 3)
            vertex_colors = mesh.visual.vertex_colors[:, :3]  # RGB only
            
            # Average the colors of the 3 vertices of each face
            colors_per_voxel = vertex_colors[face_vertices].mean(axis=1).astype(np.uint8)
            
            # Create individual colored cubes
            voxel_meshes = []
            box = trimesh.creation.box(extents=[pitch, pitch, pitch])
            
            for i, (center, color) in enumerate(zip(voxel_centers, colors_per_voxel)):
                cube = box.copy()
                cube.apply_translation(center)
                # Set vertex colors for this cube
                cube.visual.vertex_colors = np.tile(np.append(color, 255), (len(cube.vertices), 1))
                voxel_meshes.append(cube)
            
            print(f"Combining {len(voxel_meshes)} colored cubes...")
            voxel_mesh = trimesh.util.concatenate(voxel_meshes)
        else:
            # No colors, just export as boxes
            voxel_mesh = voxels.as_boxes()
        
        print(f"Saving voxel mesh to {output_path}...")
        voxel_mesh.export(output_path)
        
    return voxels

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voxelize a mesh file.")
    parser.add_argument("input", type=str, help="Input mesh file (e.g., .obj, .glb)")
    parser.add_argument("--pitch", type=float, default=0.01, help="Voxel size (pitch)")
    parser.add_argument("--output", type=str, help="Output mesh file for voxels")
    
    args = parser.parse_args()
    
    if not args.output:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_voxels.obj"
        
    voxelize(args.input, args.pitch, args.output)
