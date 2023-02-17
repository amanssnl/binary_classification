## Lion Detector

This is a deep learning based model that uses the transfer learning model VGG16 for binary classification of lion and non-lion images. The model achieves an accuracy of 92%, which makes it an effective solution for detecting lions in images.

### Data Collection

- Collecting data for this project was a challenge as there was limited data available in this field. 
- To overcome this challenge, I collected data from various sources such as YouTube videos, Kaggle, Google, and other websites. 
- This allowed me to obtain a diverse set of images that could be used to train and test the model.

### Preprocessing

- Before training the model, the data was preprocessed to ensure that it was in the correct format. 
- This involved resizing the images to a uniform size, converting them to RGB format, and normalizing the pixel values to improve training efficiency.

### Training

- The model was trained using a transfer learning approach, where the pre-trained VGG16 model was fine-tuned on the lion dataset. 
- The last layer of the model was replaced with a binary classification layer, and the entire model was re-trained on the lion dataset. 
- This approach allowed the model to learn from the features extracted by VGG16, which helped to improve the accuracy of the model.

### Results

- The lion detector model achieves an accuracy of 92%, which makes it an effective solution for detecting lions in images. 
- The model can be used for a variety of applications such as wildlife conservation, monitoring, and research.

### How to Use

- To use the lion detector model, simply clone this repository and run the `lion_detector.py` script. 
- The script takes an image as input and outputs a prediction of whether the image contains a lion or not.
