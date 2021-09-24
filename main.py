import arff  # Library https://pypi.org/project/liac-arff/. For read/write arff files
import cv2
import numpy as np
import os

# https://stackoverflow.com/questions/3430372/how-do-i-get-the-full-path-of-the-current-files-directory
curr_dir = os.path.dirname(os.path.abspath(__file__))

# Retrieve the dicc from file arff
data_file_name = "ceratocystis_test_90_10_rgb_11_fs9_res20-Indexes.arff"
data_path = curr_dir + "/data/" + data_file_name
data = arff.load(open(data_path))

# Convert dicc to list. Only the data
lst_data = list(data.values())[3]

# Convert list to numeric int array
lst_data = np.array(lst_data)
num_data = lst_data.astype(float)
num_data = np.around(num_data)
num_data = num_data.astype(int)

# Get dir name from file arff
dirDataName = data_file_name[data_file_name.index("res"):data_file_name.index("-")]
# Load imgs
img = cv2.imread('images/IMG_11.JPG')
imgLabeled = cv2.imread('images/IMG_11_Labeled.png')

cont = 1
for l in num_data:

    # Get name for imgs cut
    imgName = f'{str(cont).zfill(4)}_{dirDataName}_{l[1]}_{l[2]}_{l[-1]}'
    cont += 1

    # Img not labeled
    cropped_img = img[l[1]:l[1] + 10, l[2]:l[2] + 10]  # img[start_row:end_row, start_col:end_col]
    # Create folder, if not exists
    dirname = 'results/' + dirDataName + '/images'
    if os.path.isdir(dirname):
        pass
    else:
        os.makedirs(dirname)
    # Save img not labeled
    cv2.imwrite(os.path.join(dirname, str(imgName) + '.png'), cropped_img)
    cv2.waitKey(0)

    # Img labeled
    cropped_imgLabeled = imgLabeled[l[1]:l[1] + 10, l[2]:l[2] + 10]  # img[start_row:end_row, start_col:end_col]
    # Create folder, if not exists
    dirname = 'results/' + dirDataName + '/annotations'
    if os.path.isdir(dirname):
        pass
    else:
        os.makedirs(dirname)
    # Save img labeled
    cv2.imwrite(os.path.join(dirname, str(imgName) + '.png'), cropped_imgLabeled)
    cv2.waitKey(0)