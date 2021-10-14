import cv2
import os

# Change these values to match the desired subimage size
subImageHeight = 50
subImageWidth = 50

# Change this to the desired image to be cropped. Add the image on the 'images' folder
img = 'images/IMG_11_Labeled.png'

# Get dir name from img name
ini = 0
if img.find("/") > -1:
    ini = img.find("/") + 1

end = len(img)
if img.find(".") > -1:
    end = img.find(".")

dirDataName = img[ini:end]

# Load img
img = cv2.imread(img)

# Get img size
imageHeight = img.shape[0]
imageWidth = img.shape[1]

imgLabeled = cv2.imread('images/IMG_11_Labeled.png')

# Create folder, if not exists
dirName = 'results/' + dirDataName + '/annotations'
if os.path.isdir(dirName):
    pass
else:
    os.makedirs(dirName)

# Aux vars to name the sub-images and control the lassos
cont = 1
auxH = 0
auxW = 0

while auxH < imageHeight:
    while auxW < imageWidth:

        if (auxW + subImageWidth) > imageWidth:
            auxW = auxW + subImageWidth + 1
            continue

        # Get name for imgs cut
        imgName = f'{str(cont).zfill(4)}_{dirDataName}_{auxH}_{auxW}'
        cont += 1

        # Cropping the img
        cropped_img = img[auxH:auxH + subImageHeight,
                      auxW:auxW + subImageWidth]  # img[start_col:end_col, start_row:end_row]

        # Save img labeled
        cv2.imwrite(os.path.join(dirName, str(imgName) + '.png'), cropped_img)
        cv2.waitKey(0)

        auxW = auxW + subImageWidth + 1

    if (auxH + subImageHeight) > imageHeight:
        auxH = auxH + subImageHeight + 1
        continue

    auxH = auxH + subImageHeight + 1
    auxW = 0
