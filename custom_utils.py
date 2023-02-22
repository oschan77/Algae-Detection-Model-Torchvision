import torch
import torchvision
import torch.nn as nn
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import random


# Function to apply non-maximum suppression on the predicted bounding boxes
def apply_nms(orig_prediction, iou_thresh=0.3):
    keep = torchvision.ops.nms(
        orig_prediction['boxes'].cpu(),  # Bounding box coordinates
        orig_prediction['scores'].cpu(),  # Confidence scores for each box
        iou_thresh  # IoU threshold for suppression
    )

    return {
        'boxes': orig_prediction['boxes'].cpu()[keep],  # Filtered bounding boxes
        'scores': orig_prediction['scores'].cpu()[keep],  # Filtered confidence scores
        'labels': orig_prediction['labels'].cpu()[keep]  # Filtered class labels
    }


# Function to convert a PyTorch tensor to a PIL image
def torch_to_pil(img):
    return transforms.ToPILImage()(img).convert('RGB')


# Function to plot an image with bounding boxes
def plot_img_bbox(img, target, num_classes):
    fig, a = plt.subplots(1, 1, figsize=(10, 10))  # Create a new plot
    a.imshow(img)  # Display the image in the plot

    cmap = plt.cm.get_cmap('hsv', num_classes + 1)  # Define a color map for the boxes

    # Loop over the bounding boxes and add a rectangle to the plot for each box
    for i, (box, label) in enumerate(zip(target['boxes'], target['labels'])):
        x, y, width, height = box[0], box[1], box[2] - box[0], box[3] - box[1]

        # Define a rectangle with the appropriate coordinates and style
        rect = patches.Rectangle(
            (x, y),
            width,
            height,
            linewidth=2,
            edgecolor=cmap(label),
            facecolor='none'
        )

        a.add_patch(rect)  # Add the rectangle to the plot

    plt.show()  # Show the plot


# Function to perform inference on a random image from the dataset and plot the results
def inference_and_plot(dataset, model, device, iou_thresh, num_classes):
    model.eval()  # Set the model to evaluation mode

    with torch.inference_mode():
        idx = random.randint(0, len(dataset) - 1)  # Choose a random image index
        img, target = dataset[idx]  # Get the image and its ground truth bounding boxes
        prediction = model([img.to(device)])[0]  # Run inference on the image to get predicted boxes

    nms_prediction = apply_nms(prediction, iou_thresh)  # Apply non-maximum suppression to the predicted boxes

    plot_img_bbox(torch_to_pil(img), nms_prediction, num_classes)  # Plot the image with the filtered boxes


# Function to save a PyTorch model to a file
def save_model(model, target_path, model_name):
    assert model_name.endswith('.pth') or model_name.endswith('.pt'), \
        f'Invalid model name: {model_name}. Model name should end with ".pth" or ".pt".'

    target_path = Path(target_path)  # Convert the target path to a Path object
    target_path.mkdir(parents=True, exist_ok=True)  # Create the target directory if it doesn't exist

    torch.save(model.state_dict(), target_path / model_name) 
