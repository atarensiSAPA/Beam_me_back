import os

def validate_image_format(dir_path: str) -> bool:
    invalid_images = []

    try:
        for root, _, files in os.walk(dir_path):
            for file in files:
                image_path = os.path.join(root, file)
                if not is_valid_image(image_path):
                    invalid_images.append(image_path)

        if invalid_images:
            print("Invalid images found:")
            for img in invalid_images:
                print(f" - {img}")
                single_img = img.split("/")
                single_img = single_img[-1]
                print(f"Moving {single_img} to invalid_images folder")
                os.replace(img, "images/invalid_images/" + single_img)
            return False
        else:
            print("All images are valid.")
        return True
    except Exception as e:
        print(f"Error validating image format: {e}")
        return False

def is_valid_image(image_path: str) -> bool:
    valid_extensions = ['.jpg', '.jpeg', '.png']
    _, ext = os.path.splitext(image_path)
    return ext.lower() in valid_extensions
