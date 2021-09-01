# Dog Breed Classification

This project is for constructing algorithms for a dog identification app. The technical report is published on [Medium blog post](https://medium.com/@ollielo/how-to-construct-your-dog-breed-classifier-46688b3e123c).

## Overview

In the current project, we aim at constructing a dog breed classifier by deep learning techniques and developing a prototype of a dog identification app using the built classifier. This app is capable of reading any user-provided picture as app input. If the app detects a dog in the image, it will output its classification result, an estimation of the dog’s breed. If the app detects a human in the image, it will indicate this human resembles what kind of dog. If the app detects neither a dog nor a human in the image, it will post a message that something else was detected.

The current project involves below pipeline/steps, which I will elaborate on in the following sections of this blog post.

* Step 0: Import Datasets
* Step 1: Detect Humans
* Step 2: Detect Dogs
* Step 3: Create a CNN to Classify Dog Breeds (from Scratch)
* Step 4: Use a CNN to Classify Dog Breeds (using Transfer Learning)
* Step 5: Create a CNN to Classify Dog Breeds (using Transfer Learning)
* Step 6: Write your Algorithm
* Step 7: Test Your Algorithm

## Files

* `images`: used images in `dog_app.ipynb` and `dog_app.html`
* `dog_app.ipynb`: all methods and main algorithms used in the app
* `dog_app.html`: compiled html version of `dog_app.ipynb`

## Libraries

* tensorflow
* keras
* numpy
* matplotlib
* cv2

## Conclusion

In this project, we utilized several methods for developing a dog breed identification app. Employing CNN transfer learning, we obtained an accuracy of 83% in our testing set. Nevertheless, there is some advice for further improving our algorithm in the future:

* Try and train on different model structures. A tip to adjust model structure is to use XAI to check where the model attends to. If the model always attends to the wrong place, we can further prune the corresponding deleterious network.
* Adopt data augmentation (such as mirror flip, rotation, color adjustment, resizing, or other common computer graphics skills) to enhance the generalization power of the model, which can prevent overfitting.
* Although I have tried on different optimization algorithms and batch sizes, perhaps other choices of optimization algorithms, learning rate (including learning rate scheduling), or regularization could help.

All in all, high accuracy levels suggest that we already have a serious model that could work within a real app.
