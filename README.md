# Detection-of-fabric-defects

This project involved the implementation of convolution neural networks to distinguish fabric defects according to different type of defects from different cameras.

### Neural Network Architecture
I developed a model based on that proposed by Gao et al. in "Woven Fabric Defect Detection Based on Convolutional Neural Network for Binary Classification". However this neuronal network has been adapted to a multi-class classification to evaluate when is "Good Fabrics" but also "Fabrics with needle defects" and "Fabrics with lycra defect".

This multi-layer convolution neural network is implemented to extract the most discriminating features and identify defective tissue images.

### Dataset
The dataset consisted of 6000 images divided in 2000 Good, 2000 Needle defect and 2000 Lycra defect. Unfortunatley, the size of data is too big to be upload.

To avoid overfitting and better model generalization, data augmentation was made to increase the number of samples. The transformations were: image normalization, rotation below 15º, horizontal and vertical.

### PROJECT STILL UNDER DEVELOPMENT
The model was run twice. The first in 50 epochs and batch size 16 and the second in 35 epochs and batch size 32. In both the validation acuracy was about 97-98%. Still need to test

In the future, it would be interesting to have another new dataset with different images to further analyze the model's behavior in predicting new images. A cascading classifier would be interesting to implement in the future when it first detected if it was defective and then what type of defect it was.
