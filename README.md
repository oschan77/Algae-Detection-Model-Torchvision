# Algae-Detection-System

The code trains a Faster R-CNN object detection model using PyTorch and tracks the training progress with Weights & Biases. It includes the following steps:

1. Parse command-line arguments using the argparse library.
2. Initialize a Weights & Biases project and set configuration parameters based on the command-line arguments.
3. If multiple GPUs are available, initialize a distributed training environment using the torch.distributed library.
4. Load and split a dataset of images and annotations into training and test sets using custom functions create_datasets() and create_dataloaders().
5. Define a Faster R-CNN model using a custom function get_faster_rcnn_model().
6. Wrap the model with DistributedDataParallel if multiple GPUs are available or with DataParallel otherwise, to perform data parallelism during training.
7. Set the optimizer and loop through the number of epochs specified in the command-line arguments.
8. In each epoch, train the model using train_one_epoch() function and calculate the training loss. Log the epoch number and training loss using Weights & Biases.
9. Evaluate the model on the test set using the evaluate() function.
10. Save the trained model to a specified directory using a custom function save_model().
11. Clean up the distributed training environment if multiple GPUs are available.
