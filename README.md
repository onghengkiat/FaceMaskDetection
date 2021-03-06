# FaceMaskDetection

## About this Project
This project is about classifiying whether or not the person(s) inside the images are wearing a mask or not. It is a project for the course in my university which is WIX3001 Soft Computing

There are 2 objectives for this project which are:
1) To determine whether the person is wearing a mask or not
2) To evaluate the performance of the classification model

## Dataset Source and Reference 
https://github.com/Nikzy7/Covid-Face-Mask-Detector

## How to run
1) Install the packages required by running 

        pip install -r requirements.txt
 
2) Train the model by running 

        python3 train.py
  
3) Classify the images by running 

        python3 predict.py
 
**Notes: In order to change the images to be classified can modify on this part of codes from line 131 to line 136**

      images = [
        cv2.imread("dataset/with_mask/0-with-mask.jpg"), 
        cv2.imread("dataset/without_mask/0.jpg"), 
        cv2.imread("Ong Heng Kiat(without mask).jpg"), 
        cv2.imread("Ong Heng Kiat(with mask).jpg")
      ]

## Sample Input and Output
### Without Mask
<p align="center">
  <img src="Ong Heng Kiat(without mask).jpg" width="450" height="350">
</p>

<p align="center">
  <img src="result/Without Mask Result.png" width="450" height="350">
</p>

### With Mask
<p align="center">
  <img src="Ong Heng Kiat(with mask).jpg" width="450" height="350">
</p>

<p align="center">
  <img src="result/With Mask Result.png" width="450" height="350">
</p>
