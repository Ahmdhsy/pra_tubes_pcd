import os
import csv
import random

meta_folder = 'meta'

def create_photo_paths_csv(meta_folder, output_file="photo_paths.csv"):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["name", "photo_path"])
        
        for person in os.listdir(meta_folder):
            person_folder = os.path.join(meta_folder, person)
            if os.path.isdir(person_folder):
                for photo in os.listdir(person_folder):
                    if photo.lower().endswith(('jpg', 'png', 'jpeg')):
                        photo_path = os.path.join(person_folder, photo)
                        writer.writerow([person, photo_path])
    print(f"[INFO] {output_file} created.")

def create_pairs_csv(meta_folder, output_file="pairs.csv"):
    pairs = []
    persons = [person for person in os.listdir(meta_folder) if os.path.isdir(os.path.join(meta_folder, person))]
    
    for person in persons:
        photos = [os.path.join(meta_folder, person, photo) for photo in os.listdir(os.path.join(meta_folder, person)) if photo.lower().endswith(('jpg', 'png', 'jpeg'))]
        for i in range(len(photos)):
            for j in range(i + 1, len(photos)):
                pairs.append([photos[i], photos[j], 1])
    
    for i in range(len(persons)):
        for j in range(i + 1, len(persons)):
            person1 = persons[i]
            person2 = persons[j]
            photos1 = [os.path.join(meta_folder, person1, photo) for photo in os.listdir(os.path.join(meta_folder, person1)) if photo.lower().endswith(('jpg', 'png', 'jpeg'))]
            photos2 = [os.path.join(meta_folder, person2, photo) for photo in os.listdir(os.path.join(meta_folder, person2)) if photo.lower().endswith(('jpg', 'png', 'jpeg'))]
            random.shuffle(photos2)
            for photo1 in photos1:
                for photo2 in photos2:
                    pairs.append([photo1, photo2, 0])
    
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["photo1", "photo2", "label"])
        writer.writerows(pairs)
    print(f"[INFO] {output_file} created.")

def main():
    create_photo_paths_csv(meta_folder)
    create_pairs_csv(meta_folder)

if __name__ == "__main__":
    main()
