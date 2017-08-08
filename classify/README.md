### Image Classification - Ten butterfly species:
Danaus Plexippus

Heliconius Charitonius

Heliconius Erato

Junonia Coenia

Lycaena Phlaeas

Nymphalis Antiopa

Papilio Cresphontes

Pieris Rapae

Vanessa Atalanta

Vanessa Cardui


### Prerequisites
1. Download VGG16, VGG19, Inception V3, Xception trained models from https://drive.google.com/drive/folders/0B5V_P8iI1LUeMzBTQzlWLTZwVTg
2. Store the candidate butterfly image or images in the images folder
3. Install Keras Library version 1.2.2
4. Install Tensorflow Library version 0.12

### Usage
python classify_butterfly_image.py --image images/butterfly.jpg --model inception

### Disclaimer
Since these models are specifically trained to classify the ten species listed above it is likely that if an image of a cat, say for example, is used for classification, the models may try to fit the image into one of the ten categories. The user is encouraged to provide candidate images that ONLY reflect any of these ten species. The Web has a wealth of images available for use to test these models out.
