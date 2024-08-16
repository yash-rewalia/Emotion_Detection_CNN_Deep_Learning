### Emotion Detection Using CNN
Emotion detection is crucial for enhancing human-computer interaction, making technology more responsive and empathetic to users' emotional states. It plays a vital role in mental health monitoring by identifying emotional cues that may indicate conditions like depression or anxiety, enabling timely intervention. Additionally, emotion detection has applications in personalized marketing, customer service, and education, where understanding emotions can lead to more tailored and effective experiences. By bridging the gap between human emotions and machine understanding, emotion detection paves the way for more intuitive and impactful technological solutions.

![image](https://drive.google.com/file/d/1T9jB9OZYvR0nyg20pPIBxhQ-Nymbd8oR/view?usp=sharing)

## Overview
This project focuses on detecting emotions from images using Convolutional Neural Networks (CNN). The task involves building and comparing the performance of four different models:

1. Custom CNN: A CNN model designed from scratch for emotion detection.

2. Custom CNN with Data Augmentation: The same custom CNN model, but with data augmentation techniques applied to improve generalization.

3. Transfer Learning with VGG16: Utilizing the pre-trained VGG16 model to leverage transfer learning for emotion detection.

4. ResNet50v2: Another approach using the pre-trained ResNet50v2 model for transfer learning.

## Dataset
The dataset used in this project consists of labeled images representing different emotions. The images are preprocessed and split into training, validation, and test sets.
link for dataset:- https://www.kaggle.com/datasets/msambare/fer2013

## Models
1. Custom CNN
A custom CNN architecture built from scratch. It includes convolutional layers, pooling layers, and fully connected layers. The model is trained on the dataset and evaluated on unseen data.

2. Custom CNN with Data Augmentation
This model builds upon the custom CNN by incorporating data augmentation techniques such as rotation, zoom, and flipping. This helps the model generalize better to new data.

3. Transfer Learning with VGG16
This approach leverages the pre-trained VGG16 model, which has been trained on a large dataset (ImageNet). The top layers are fine-tuned to adapt to the emotion detection task.

4. ResNet50v2
ResNet50v2 is another pre-trained model used for transfer learning. Its residual connections help in training deep networks more effectively. The model is fine-tuned for emotion detection.

## Evaluation
Each model is evaluated on the test set using accuracy, precision, recall, and F1-score. The performance of the models is compared, and the results are documented.

## Installation
To run this project, ensure you have the following dependencies installed:

1. Python 3.10
2. TensorFlow/Keras
3. NumPy
4. OpenCV
5. Matplotlib
6. scikit-learn

## Contributing
Feel free to contribute to this project by submitting issues or pull requests. Any contributions, whether it's improving the model or enhancing the documentation, are welcome.

## Contact
For any inquiries or further information, please contact Yash Kumar at yash.krewalia@gmail.com.
