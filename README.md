![Header Graphic](https://github.com/thegrandblooms/Phase_4_Project_FIS-DS/blob/c1442bc8111b37945f3ec61caba1b970ff6689fc/graphics/header.jpg)
# Recommendation Systems - Flatiron Data Science, Phase 4 Project
Author: Blake McMeekin

## Overview
Pneumonia is a potentially life-threatening ["infection that inflames the air sacs in one or both lungs."](https://www.mayoclinic.org/diseases-conditions/pneumonia/symptoms-causes/syc-20354204#:~:text=Pneumonia%20is%20an%20infection%20that,and%20fungi%2C%20can%20cause%20pneumonia.) "A variety of organisms, including bacteria, viruses and fungi can cause pneumonia." Pneumonia is especially relevant today in the treatment of serious Covid-19 infections. 

In this notebook we'll build a deep-learning model to analyze x-ray images for the presence of Pneumonia. We'll be looking at over 5000 X-rays of children age 1-5 sourced from [UCSD](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia), building a few convolutional neural nets to interpret and classify these images. Note that in order for our model to generalize to adults, we would need additional training data, which might be supplemented by the model built in this notebook as pretraining.

We'll use data generation to increase the size of this dataset, helping our model generalize to new images or images at different angles or scales. This may make our model take longer to train, but increases performance on unseen data. Image resolution a balance between model performance and training times. Since all of the training in this notebook was done locally, an image size of 100x100 was chosen to keep training times under 1 hour.

The packages we use for this analysis are standard for deep learning, with many keras and tensorflow packages for deep learning and sklearn packages for model evaluation and data preparation. These packages are super deep and give us more than enough to work with for this stage of prototyping.

Our model architecture is iterative, starting with a simple three-layer CNN. We quickly see the limits of this simple system, and have three more iterations of increasing complexity, the second model having 9 layers and dropout, with the third model having ten layers with dropout and a learning rate, and the fourth and final model has eleven layers with dropout and a learning rate. Our final model had an accuracy of 92% on holdout test data, with a particularly nice 95% recall on sick cases. This means only 1 in 20 unhealthy x-rays are missed. This could be further tuned by experimenting with the decision threshold if a higher precision or recall were needed in application.

## Organizational Problem

Increasing the detection rate of pneumonia can increase the quality of healthcare in hospitals and clinics around the world, and automated processes can help particularly when hospitals are understaffed or short of funding. One machine learning model could analyze thousands of x-rays per hour, serving clinics across the globe.

![Floral image of lungs](https://github.com/thegrandblooms/Phase_4_Project_FIS-DS/blob/c1442bc8111b37945f3ec61caba1b970ff6689fc/graphics/DALL%C2%B7E%202022-09-09%2022.47.43%20-%20lungs%20made%20of%20a%20flower%20arrangement,%20poignant%20and%20thought-provoking%20digital%20art%20photography.png)

## Data

Our data consists of 5846 X-ray images of lungs of children aged 1-5 provided by UCSD. In addition to this, we use data generation to increase the size of this dataset by rotating or rescaling images. Our training data also has a considerable class imbalance, with about three times as many unhealthy lungs as healthy lungs.

![Images of healthy/unhealthy lungs](https://github.com/thegrandblooms/Phase_4_Project_FIS-DS/blob/c1442bc8111b37945f3ec61caba1b970ff6689fc/graphics/lungs.png)

## Methods

Our analysis consists of three stages:
- 1.) Data Cleaning and Processing
  - Visual inspection, file preparation, data generation, rescaling, etc
- 2.) Deep Learning
  - CNN, model iterations and validation, hyperparameter tuning
- 3.) Validation and Implementation
  - Validation and model comparison, final model selection, and presentation

## Validation

For validating our model, we split our images into train, validation and test sets. The original data has only 16 validation images - we change this to a more traditional 80/20 split, and use that validation set to see how well our model performs on about 1000 "less seen" images. Once we have a final model built, we'll see how it performs on our holdout test data - about 600 images that the model has absolutely never seen.

## Findings

![Confusion Matrix](https://github.com/thegrandblooms/Phase_4_Project_FIS-DS/blob/c1442bc8111b37945f3ec61caba1b970ff6689fc/graphics/confusion_matrix.png)

- Test accuracy around 92%
- Can inspect thousands of cases per hour
- Extremely affordable compared to human inspection
- Can double-check human diagnoses

## Conclusion

In my opinion, deep learning can absolutely provide value in medical diagnosis. Double-checking images that are being inspected by doctors could increase accuracy or highlight interesting zones in images - more images could be inspected for less expense and workload, or other types of images could be inspected and categorized at scale. At this level of accuracy there are already tons of images, and all of the models in this presentation were trained on a local desktop in under an hour - better performance in a production model seems very likely.

## Next Steps

Process Integration:
Talking to target users and supporting them with a simple UI, or centralizing image processing by taking in images from many medical facilities

More computation or Data:
It’s likely that these models can continue to be improved with more computation and X-Ray Images

More X-Ray Machine Learning:
Can broken bones be identified? What about tumors or cancers? What do doctors need help with? Maybe machine learning can help.

## For more information

See the full analysis in the Jupyter Notebook or review the presentation in the pdf.

For additional information, contact Blake McMeekin at blakemcme@gmail.com

## Repository Structure

```
├── data
│   ├── test
│   ├── train
│   └── val
├── graphics
├── Pneumonia Detection Presentation.pdf
├── README.md
└── CNN for Pneumonia Detection.ipynb
```
