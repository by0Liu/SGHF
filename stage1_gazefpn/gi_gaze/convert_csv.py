import os
import cv2
import json
import numpy as np

#input csv file
csv_path = './data/12024-3-25-14-38.csv'
#folder of the images corresponding to csv
source_image_path = './data/val/Type 3/images/'#benign, malignant

#path to export files
export_path = './data/val/Type 3/'

#some files are missing in the csv
source_files = os.listdir(source_image_path)

#interpolation method
interp =cv2.INTER_CUBIC

#the image size on screen should be as the same as this one
gaze_map_height = 900

#tune these
kernel_size = 199 #has to be an odd value
sigma = 50

#from the 'Eye tracking based deep learning analysis for the early detection of diabetic retinopathy: A pilot study'
#kernel_size = 99
#sigma = 55

#from the 'Follow My Eye: Using Gaze to Supervise Computer-Aided Diagnosis' paper NOT working well
#kernel_size = 99
#sigma = 30.2

#make export folders
if not os.path.exists(export_path+'images'):
    os.makedirs(export_path+'images')
if not os.path.exists(export_path+'attentions'):
    os.makedirs(export_path+'attentions')
if not os.path.exists(export_path+'heatmaps'):
    os.makedirs(export_path+'heatmaps')

#feom the source code of gaze tracker
def heatmapColormap(image: np.ndarray, code) -> np.ndarray:
    r"""The Input Image should be single channel 0-1 float.
    The output is a 3-channel BGR(Not RGB) 0-255 uint8 image.
    """
    return cv2.applyColorMap((image * 255).astype(np.uint8), code)


def superimposeHeatmapToImage(heatmap: np.ndarray, image: np.ndarray) -> np.ndarray:
    assert heatmap.shape[:2] == image.shape[:2]
    return cv2.addWeighted(image, 0.7, heatmap, 0.3, 0)


def pointToHeatmap(pointList, normalize=True, heatmapShape=(800, 800)):
    canvas = np.zeros(heatmapShape)
    for p in pointList:
        if p[1] <= heatmapShape[0] and p[0] <= heatmapShape[1]:
            canvas[p[1]][p[0]] = 1
    g = cv2.GaussianBlur(canvas, ksize=(kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)
    if normalize:
        g = cv2.normalize(g, None, alpha=0, beta=1,
                          norm_type=cv2.NORM_MINMAX)
    return g

def get_width(img_width, img_height):
    ratio = (gaze_map_height / float(img_height))
    wSize = int((float(img_width) * float(ratio)))
    return wSize

def lineProcess(line: str):
    info = line.split(';')
    filename = info[0]
    #groundTruth = filename.split('\\')[-2]
    #annotation = info[1]
    gaze = json.loads(info[2])
    imgName = filename.split('\\')[-1]
    #if the file in csv not found in the source folder
    if imgName not in source_files:
        print('missing:'+source_image_path+imgName)
        return
    img_source = cv2.imread(source_image_path+imgName) #h, w, c
    img_shape = img_source.shape
    gaze_map_width = get_width(img_shape[1], img_shape[0])
    img = cv2.resize(img_source, (gaze_map_width, gaze_map_height), interpolation=interp) #resize need w, h rather than h, w
    heatmap = pointToHeatmap(gaze, heatmapShape=(gaze_map_height, gaze_map_width)) #heatmap need h, w
    vis_heatmap = superimposeHeatmapToImage(heatmap=heatmapColormap(heatmap, cv2.COLORMAP_JET),
                                    image=img)
    vis_heatmap = cv2.resize(vis_heatmap, (img_shape[1], img_shape[0]), interpolation=interp)
    attention = cv2.resize(heatmap*255, (img_shape[1], img_shape[0]), interpolation=interp).astype(np.uint8)
    cv2.imwrite(export_path+'heatmaps/'+imgName.replace('.jpg', '.png'), vis_heatmap)
    cv2.imwrite(export_path+'attentions/'+imgName.replace('.jpg', '.tif'), attention)
    cv2.imwrite(export_path+'images/'+imgName.replace('.jpg', '.png'), img_source)


for l in open(csv_path).read().split('\n'):
    if l != '':
        lineProcess(l)


