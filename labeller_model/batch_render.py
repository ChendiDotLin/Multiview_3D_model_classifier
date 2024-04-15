import subprocess
import os

# Folder containing the objects
object_folder = "training_data_models"

# List all files in the folder
object_files = os.listdir(object_folder)

# Iterate over the list of objects
for obj in object_files:
    # Command to run Blender with the specified options
    command = [
        "/Applications/Blender.app/Contents/MacOS/Blender",
        "-b",
        "-P",
        "blender_script.py",
        "--",
        "--object_path",
        os.path.join(object_folder, obj),
        "--output_dir",
        "./views",
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
