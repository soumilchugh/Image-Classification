# Image-Classification
* This is the most basic implementation of transfer learning for image classification on CIFAR 10 dataset.<br /> 
* VGG-16 is one of most popular CNN network trained on ImageNet for image classification<br />
* Using pretrained weights from the VGG-16 model, we can retrain part of the VGG16 network with our own dataset to perform image classification. <br />
* We first restore the weights of the some of the layers of VGG-16 and add our own layers on top of it to perform classification on CIFAR 10 dataset<br />
* CIFAR 10 dataset is 32x32 and is resized to 48x48 since VGG-16 can handle this input size.<br />
* To convert the 2D feature maps from the CNN layers to 1D, a global max pooling operation is used instead of flattening the 2D tensor since this has shown to work well recently<br />

Instructions to run the experiment<br />
* Download the VGG-16 weights and the cifar dataset files from the following google drive link<br />
https://drive.google.com/open?id=1F11GFMgB3cKLvQdxSiM96V0BH76Mqtfa <br />
* Place these files in the same folder as the main.py file<br />
* Run the main.py script and the checkpoint will be saved once the model is trained for a certain number of epochs
