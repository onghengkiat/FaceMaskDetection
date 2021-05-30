# FaceMaskDetection

## About this Project

## Dataset Source and Reference 
https://github.com/Nikzy7/Covid-Face-Mask-Detector

## How to run
1) Install the packages required by running 

  pip install -r requirements.txt
 
2) Train the model by running 

  python3 train.py
  
3) Classify the images by running 

  python3 predict.py
 
**Notes: In order to change the images to be classified can modify on this part of codes from line 130 to line 136**

  # Image files to be classified
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
  <img src="Without Mask Result.png" width="450" height="350">
</p>

### With Mask
<p align="center">
  <img src="Ong Heng Kiat(with mask).jpg" width="450" height="350">
</p>

<p align="center">
  <img src="With Mask Result.png" width="450" height="350">
</p>
