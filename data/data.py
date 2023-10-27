import tarfile
import os
import shutil
import pandas as pd


path_to_data = 'C:/Users/ensin/OneDrive/Documenten/Universiteit/y2b1/Research Topics/project/waterbird_complete95_forest2water2'
waterbirds_complete = 'C:/Users/ensin/OneDrive/Documenten/Universiteit/y2b1/Research Topics/project/waterbird_complete95_2class'


def organize_bird_images(csv_file, image_dir, output_dir):
    # Create output directories for landbirds and waterbirds
    landbird_dir = os.path.join(output_dir, 'landbirds')
    waterbird_dir = os.path.join(output_dir, 'waterbirds')

    os.makedirs(landbird_dir, exist_ok=True)
    os.makedirs(waterbird_dir, exist_ok=True)

    # Read the CSV file with image labels
    df = pd.read_csv(csv_file)

    for index, row in df.iterrows():
        image_filename = row['img_filename']
        label = row['y'] 

        image_name = os.path.basename(image_filename)

        # Define the destination folder based on the label
        if label == 0:
            destination = os.path.join(landbird_dir, image_name)
        elif label == 1:
            destination = os.path.join(waterbird_dir, image_name)
        else:
            print(f"Invalid label for {image_name}. Skipping.")
            continue

        source_path = os.path.join(image_dir, image_filename)

        # Copy or move the image to the appropriate folder
        shutil.copy(source_path, destination)  # Use shutil.move() to move instead of copy

    print("Organized images into 'landbirds' and 'waterbirds' folders.")


organize_bird_images('C:/Users/ensin/OneDrive/Documenten/Universiteit/y2b1/Research Topics/project/waterbird_complete95_forest2water2/metadata.csv', path_to_data, waterbirds_complete)