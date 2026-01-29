# Creates the image taxonomy file.
# Format: Image path : Image Label Prompt
import os
import csv
import random
import re


def create_image_triplets(image_path):

    # print(os.listdir(image_path))
    # for root, dirs, files in os.walk(image_path):
    #     for name in files:
    #         if name.startswith("_") or name.startswith("._"):
    #             path = os.path.join(root, name)
    #             print("Deleting:", path)
    #             os.remove(path)

    categories = os.listdir(image_path)
    categories_sampled = random.sample(categories, k=25)

    with open('./data/birds/birds.taxo', 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')

        for category in categories_sampled:
            all_images = os.listdir(os.path.join(image_path, category))

            for image in all_images:
                image_absolute_path = os.path.join(image_path, category, image)

                category_cleaned = re.sub(r'^\d+\.', '', category)
                writer.writerow([image_absolute_path, category_cleaned])
