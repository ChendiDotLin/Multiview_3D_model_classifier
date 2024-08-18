import subprocess
import os

import shutil
import multiprocessing

def flatten_directory(source_directory, target_dir):
    # Walk through the source directory
    for root, dirs, files in os.walk(source_directory):
        for file in files:
            # Check if the file is a .glb file
            if file.endswith(".glb"):
                # Construct the full file path
                file_path = os.path.join(root, file)
                # Define the target path for the .glb file
                target_path = os.path.join(target_dir, file)
                # Move the file
                shutil.move(file_path, target_path)



def render_object(target_path, output_dir):
    command = [
        "/Applications/Blender.app/Contents/MacOS/Blender",
        "-b",
        "-P",
        "persistent_blender_script.py",
        "--",
        "--object_path",
        target_path,
        "--output_dir",
        output_dir,
        "--engine",
        "CYCLES",
        "--scale",
        "0.8",
        "--num_images",
        "40",
        "--camera_dist",
        "1.2",
    ]

    # Run the Blender command
    subprocess.run(command)

def render_batch(object_files, target_path, output_dir):
    """Runs Blender to render a batch of objects."""
    command = [
        "/Applications/Blender.app/Contents/MacOS/Blender",
        "-b",
        "-P",
        "persistent_blender_script.py",
        "--",
        "--object_path",
        target_path,
        "--output_dir",
        output_dir,
        "--engine",
        "CYCLES",
        "--scale",
        "0.8",
        "--num_images",
        "40",
        "--camera_dist",
        "1.2",
    ]

    env = os.environ.copy()
    env['OBJECT_BATCH'] = ','.join(object_files)
    subprocess.run(command, env=env)

def split_batches(object_files, num_batches):
    """Split the list of object files into smaller batches."""
    batch_size = len(object_files) // num_batches
    return [object_files[i:i + batch_size] for i in range(0, len(object_files), batch_size)]
    
if __name__ == "__main__":
    # Specify the root directory
    cwd = os.getcwd()
    root_dir = cwd + "/objaverse/hf-objaverse-v1"
    target_path = cwd + "/objaverse_models"
    os.makedirs(target_path, exist_ok=True)
    # Flatten the directory
    flatten_directory(root_dir, target_path)

    # Folder containing the objects
    # object_folder = "training_data_models"

    # List all files in the folder
    output_dir = cwd + "/views"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the number of processes (adjust this based on your CPU/GPU resources)
    num_processes = multiprocessing.cpu_count()
    
    # List all files in the folder
    object_files = [f for f in os.listdir(target_path) if f.endswith('.glb')]
    # Split the list of object files into smaller batches
    batches = split_batches(object_files, num_processes)

    # Use multiprocessing to render each batch in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.starmap(render_batch, [(batch, target_path, output_dir) for batch in batches])
    # render_object(target_path, output_dir)

    # Command to run Blender with the specified options