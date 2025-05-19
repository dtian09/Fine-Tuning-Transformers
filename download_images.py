from datasets import load_dataset, load_from_disk
from PIL import Image
import os

# Create directory to store test images
os.makedirs("test_images", exist_ok=True)

if os.path.isdir("flickr30k_testset"):
    testset = load_from_disk("flickr30k_testset")
    index = 0  # Change to save a different image
    example = testset[index]
else:  
    # Load Flickr30k dataset
    testset = load_dataset("nlphuji/flickr30k", split="test", keep_in_memory=False)

    # Filter for actual test split entries (Flickr30k stores all splits in one set)
    testset = testset.filter(lambda x: x["split"] == "test", keep_in_memory=False)
    # Filter by internal 'split' field and keep only 'caption' and 'image'
    testset = testset.remove_columns(
                [col for col in testset.column_names if col not in {"caption", "image"}]
    )

    testset.save_to_disk("flickr30k_testset")
    # Choose one or more samples to save
    index = 0  # Change to save a different image
    example = testset[index]

# Get the image and caption
image: Image.Image = example["image"]  # PIL Image
caption = example["caption"]
filename = f"test_images/image_{index}.jpg"

# Save the image
image.save(filename)

print(f"Saved image to: {filename}")
print(f"Caption: {caption}")
