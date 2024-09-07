import os


def find_uids_with_40_screenshots(root_folder, output_txt):
    # List to store UIDs that meet the condition
    valid_uids = []

    # Traverse the root folder
    for uid_folder in os.listdir(root_folder):
        uid_path = os.path.join(root_folder, uid_folder)

        # Check if the path is a folder
        if os.path.isdir(uid_path):
            # Count the number of .png files in the folder
            png_files = [f for f in os.listdir(uid_path) if f.endswith(".png")]

            # Check if there are exactly 40 .png files
            if len(png_files) == 40:
                valid_uids.append(uid_folder)

    # Write the UIDs to a TXT file
    with open(output_txt, "w") as txtfile:
        for uid in valid_uids:
            txtfile.write(f"{uid}\n")

    print(
        f"Finished! UIDs with exactly 40 PNG screenshots have been written to {output_txt}"
    )


# Usage
root_folder = "/path/to/root/folder"  # Replace with the actual path to your root folder
output_txt = "output_uids.txt"  # Output TXT file name
find_uids_with_40_screenshots(root_folder, output_txt)
