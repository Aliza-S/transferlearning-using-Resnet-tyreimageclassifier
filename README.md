
Tyre Quality Classification using PyTorch This project explores the classification of digital images of vehicle tyres into 'good' and 'defective' categories using deep learning. Two different approaches were implemented and analyzed using the PyTorch framework:

A Simple Convolutional Neural Network (CNN) built from scratch.

Transfer Learning using a pre-trained ResNet-50 model.

The primary goal was to compare the effectiveness of these two methods and demonstrate key techniques for training and optimizing deep learning models for image classification.

Dataset The dataset consists of digital images of tyres, organized into two class folders:

/good/: Images of tyres in good condition.

/defective/: Images of tyres with various defects.

A standard split was used to create training and validation sets, allowing the models to be properly trained and evaluated.

Models & Methodology
Data Loading & Preprocessing
-  Dataset:  `datasets/Digital images of defective and good condition tyres`
-  Preprocessing Pipeline: 
  -  Resize:  All images are resized to 256×256
  -  Center Crop:  Cropped to 224×224 for consistency
  -  Tensor Conversion:  Images converted to PyTorch tensors
  -  Normalization:  Standardized using ImageNet mean & std  
     (mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]) 
Model Architecture
A custom   Convolutional Neural Network (CNN)   built with `torch.nn`:
-   Convolutional Layers:   Learn local spatial features
-   Activation Function:   ReLU for non-linearity
-   Pooling:   MaxPooling for spatial downsampling
-   Fully Connected Layers:   Flattened features → Dense layers
-   Output Layer:   2 neurons (Good / Defective) with Softmax

> This architecture is simple and can be extended with pretrained backbones like ResNet, VGG, or EfficientNet.

 Training Procedure
-   Loss Function:   `nn.CrossEntropyLoss` (suitable for 2-class classification)
-   Optimizer:   Adam optimizer for faster convergence
-   Device:   GPU (CUDA) is used if available, else CPU
-   Epochs:   Configurable, typically 10–20
  

 Evaluation
-   Validation Set:   A portion of the dataset is used for validation
-   Metrics:  
  - Accuracy
-   Visualizations:  
  - Accuracy/Loss curves
  - Confusion Matrix to evaluate per-class performance

Results
The trained model distinguishes between good and defective tyres, supporting quality control automation. 
The model demonstrates strong learning behavior with consistent improvements in both training and validation accuracy. By Epoch 6, it achieves 98.65% training accuracy and 95.15% validation accuracy, indicating effective generalization. While validation loss shows minor fluctuations, the overall trend supports robust performance. 
feature to be added
- Data augmentation (RandomCrop, HorizontalFlip)


 ##  New Feature: Data Augmentation

This update adds data augmentation to improve model generalization. Techniques include:
- Random horizontal flip
- Rotation
- Color jitter
- Random resized crop

Implemented using `torchvision.transforms`.
This is train loss and train accuracy before merging augmentation : Train loss: 0.22522228788734683 Train acc: 0.8976430976430977

Changes observed after merging: Train loss:0.25305921311401985 Train acc: 0.8929292929292929
Authors:

Aliza Sheikh

Sarah Fatima Islam


