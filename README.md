# Digit Recognition With Neural Network

Digit recognition is foundational in many real-world applications, from banking systems to automated document processing. The MNIST dataset, a benchmark in the field of machine learning, provides 28x28 grayscale images of handwritten digits (0-9), split into 60,000 training and 10,000 test samples.

This project employs a Convolutional Neural Network (CNN) to classify these digits. CNNs are particularly well-suited for image recognition tasks due to their ability to capture spatial hierarchies through convolutional layers. To further enhance performance, techniques such as Dropout and Batch Normalization were applied.

The CNN was designed with multiple layers for feature extraction and classification.
The layers included: 
1. Input Layer: Accepts images of shape `28x28x1`.
2. Convolutional Layers: Two layers with filters of size (3x3). Convolutional layers apply filters to extract spatial features, such as edges and textures.
3. MaxPooling Layers: Reduce spatial dimensions. MaxPooling reduces feature map dimensions, retaining critical patterns while lowering computational costs.
4. Flatten Layer: Converts 2D data into a 1D array.
5. Dense Layer: Fully connected layer with 128 neurons.
6. Dropout Layer: Prevents overfitting.
7. Output Layer: Softmax activation for 10-class classification. The Softmax activation function in the output layer assigns probabilities to each class, ensuring that the sum equals 1.
