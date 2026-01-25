import subprocess
import argparse
import os
import sys

def run_workflow(image_path, output_dir="output/workflow", pitch=0.02):
    print(f"=== Starting Workflow for {image_path} ===")
    
    # 1. Run TripoSR
    print("\n--- Step 1: Image to 3D (TripoSR) ---")
    tripo_cmd = [
        sys.executable, "run.py", 
        image_path, 
        "--output-dir", output_dir
    ]
    # Set environment variable for MPS fallback
    env = os.environ.copy()
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    
    result = subprocess.run(tripo_cmd, env=env)
    if result.returncode != 0:
        print("Error in Step 1")
        return
        
    # TripoSR saves to output_dir/0/mesh.obj
    mesh_path = os.path.join(output_dir, "0", "mesh.obj")
    if not os.path.exists(mesh_path):
        print(f"Mesh not found at {mesh_path}")
        return
        
    # 2. Voxelize
    print("\n--- Step 2: Voxelization ---")
    voxel_out = os.path.join(output_dir, "0", "mesh_voxels.obj")
    voxel_cmd = [
        sys.executable, "voxelize.py",
        mesh_path,
        "--pitch", str(pitch),
        "--output", voxel_out
    ]
    
    result = subprocess.run(voxel_cmd)
    if result.returncode != 0:
        print("Error in Step 2")
        return
        
    print(f"\n=== Workflow Complete! ===")
    print(f"Results are in: {os.path.join(output_dir, '0')}")
    print(f"Voxel mesh: {voxel_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image to Voxel Workflow")
    parser.add_argument("image", type=str, help="Input image path")
    parser.add_argument("--output-dir", type=str, default="output/workflow", help="Output directory")
    parser.add_argument("--pitch", type=float, default=0.02, help="Voxel size")
    
    args = parser.parse_args()
    
    # Ensure we are using the .venv/bin/python3
    python_exe = sys.executable
    
    run_workflow(args.image, args.output_dir, args.pitch)
