from PIL import Image
import os

def visualise_resize(image_path, resize_value, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image = Image.open(image_path)
    resized_image = image.resize((resize_value, resize_value))
    resized_image.save(output_path)


if __name__ == "__main__":
    #visualise_resize("dataset/chihuahua/img_4_799.jpg", 32, "static/chihuahua/img_4_799_32.jpg")
    #visualise_resize("dataset/muffin/img_0_187.jpg", 32, "static/muffin/img_4_880_32.jpg")
