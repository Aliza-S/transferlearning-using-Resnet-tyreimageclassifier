
Tyre Quality Classification using PyTorch This project explores the classification of digital images of vehicle tyres into 'good' and 'defective' categories using deep learning. Two different approaches were implemented and analyzed using the PyTorch framework:

A Simple Convolutional Neural Network (CNN) built from scratch.

Transfer Learning using a pre-trained ResNet-50 model.

The primary goal was to compare the effectiveness of these two methods and demonstrate key techniques for training and optimizing deep learning models for image classification.

Dataset The dataset consists of digital images of tyres, organized into two class folders:

/good/: Images of tyres in good condition.

/defective/: Images of tyres with various defects.

A standard split was used to create training and validation sets to properly train and evaluate the models.

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
-   Model Saving:   Final model weights are stored as `model.pkl`

 Evaluation
-   Validation Set:   A portion of the dataset is used for validation
-   Metrics:  
  - Accuracy
  - (Optional) Precision, Recall, F1-score for class-level analysis
-   Visualizations:  
  - Accuracy/Loss curves
  - Confusion Matrix to evaluate per-class performance

Results
The trained model distinguishes between good and defective tyres, supporting quality control automation. 
feature to be added
- Data augmentation (RandomCrop, HorizontalFlip)
Authors
Aliza Sheikh
Sarah Fatima Islam


