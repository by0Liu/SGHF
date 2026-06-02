<<<<<<< HEAD
import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
import csv
import pathlib


eyetracking_dataset_path = './data/reflacx/main_data/'

def get_gaussian(y, x, sy, sx, sizey, sizex, shown_rects_image_space):
    mu = [y - shown_rects_image_space[1], x - shown_rects_image_space[0]]
    sig = [sy ** 2, sx ** 2]
    x = np.arange(0, shown_rects_image_space[2] - shown_rects_image_space[0], 1)
    y = np.arange(0, shown_rects_image_space[3] - shown_rects_image_space[1], 1)
    X, Y = np.meshgrid(x, y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 1] = X
    pos[:, :, 0] = Y
    to_return = np.zeros([sizey, sizex])

    to_return[int(round(shown_rects_image_space[1])):int(round(shown_rects_image_space[3])),
    int(round(shown_rects_image_space[0])):int(round(shown_rects_image_space[2]))] = multivariate_normal(mu, sig).pdf(pos)

    return to_return


def create_heatmap(sequence_table, size_x, size_y):
    img = np.zeros((size_y, size_x), dtype=np.float32)
    for index, row in sequence_table.iterrows():
        angle_circle = 1
        shown_rects_image_space = [round(row['xmin_shown_from_image']), round(row['ymin_shown_from_image']),
                                   round(row['xmax_shown_from_image']), round(row['ymax_shown_from_image'])]
        gaussian = get_gaussian(row['y_position'], row['x_position'],
                                row['angular_resolution_y_pixels_per_degree'] * angle_circle,
                                row['angular_resolution_x_pixels_per_degree'] * angle_circle, size_y, size_x,
                                shown_rects_image_space)

        img += gaussian * (row['timestamp_end_fixation'] - row['timestamp_start_fixation'])
    return img / np.sum(img)


def create_heatmaps(data_folder, filename_phase, folder_name='heatmaps', phase=None):
    df = pd.read_csv(data_folder + filename_phase)
    df = df[df['eye_tracking_data_discarded'] == False]
    all_images = df['image'].unique()
    for trial, image_name in enumerate(sorted(all_images)):
        df_this_trial = df[df['image'] == image_name]
        index_image = 0
        for _, df_this_index_this_trial in df_this_trial.iterrows():
            print('trial', trial, 'index_image', index_image)
            image_size_x = int(float(df_this_index_this_trial['image_size_x']))
            image_size_y = int(float(df_this_index_this_trial['image_size_y']))
            fixations = pd.read_csv(f'{data_folder}/{df_this_index_this_trial["id"]}/fixations.csv')
            this_img = create_heatmap(fixations, image_size_x, image_size_y)  # 生成的热图

            info_dict = {'np_image': this_img, 'img_path': image_name, 'trial': trial,
                         'id': df_this_index_this_trial['id'], 'phase': phase}
            image_name = image_name.replace("/", "")
            save_dir = pathlib.Path(folder_name) / image_name
            save_dir.mkdir(parents=True, exist_ok=True)
            np.save(save_dir / "eye_cam.npy", info_dict)
            index_image += 1


file_phase = 'metadata_phase_3.csv'
create_heatmaps(
    eyetracking_dataset_path,
    file_phase,
    folder_name='./data/cxr_gaze/eye_cam',
    phase=1
=======
import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
import csv
import pathlib


eyetracking_dataset_path = './data/reflacx/main_data/'

def get_gaussian(y, x, sy, sx, sizey, sizex, shown_rects_image_space):
    mu = [y - shown_rects_image_space[1], x - shown_rects_image_space[0]]
    sig = [sy ** 2, sx ** 2]
    x = np.arange(0, shown_rects_image_space[2] - shown_rects_image_space[0], 1)
    y = np.arange(0, shown_rects_image_space[3] - shown_rects_image_space[1], 1)
    X, Y = np.meshgrid(x, y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 1] = X
    pos[:, :, 0] = Y
    to_return = np.zeros([sizey, sizex])

    to_return[int(round(shown_rects_image_space[1])):int(round(shown_rects_image_space[3])),
    int(round(shown_rects_image_space[0])):int(round(shown_rects_image_space[2]))] = multivariate_normal(mu, sig).pdf(pos)

    return to_return


def create_heatmap(sequence_table, size_x, size_y):
    img = np.zeros((size_y, size_x), dtype=np.float32)
    for index, row in sequence_table.iterrows():
        angle_circle = 1
        shown_rects_image_space = [round(row['xmin_shown_from_image']), round(row['ymin_shown_from_image']),
                                   round(row['xmax_shown_from_image']), round(row['ymax_shown_from_image'])]
        gaussian = get_gaussian(row['y_position'], row['x_position'],
                                row['angular_resolution_y_pixels_per_degree'] * angle_circle,
                                row['angular_resolution_x_pixels_per_degree'] * angle_circle, size_y, size_x,
                                shown_rects_image_space)

        img += gaussian * (row['timestamp_end_fixation'] - row['timestamp_start_fixation'])
    return img / np.sum(img)


def create_heatmaps(data_folder, filename_phase, folder_name='heatmaps', phase=None):
    df = pd.read_csv(data_folder + filename_phase)
    df = df[df['eye_tracking_data_discarded'] == False]
    all_images = df['image'].unique()
    for trial, image_name in enumerate(sorted(all_images)):
        df_this_trial = df[df['image'] == image_name]
        index_image = 0
        for _, df_this_index_this_trial in df_this_trial.iterrows():
            print('trial', trial, 'index_image', index_image)
            image_size_x = int(float(df_this_index_this_trial['image_size_x']))
            image_size_y = int(float(df_this_index_this_trial['image_size_y']))
            fixations = pd.read_csv(f'{data_folder}/{df_this_index_this_trial["id"]}/fixations.csv')
            this_img = create_heatmap(fixations, image_size_x, image_size_y)  # 生成的热图

            info_dict = {'np_image': this_img, 'img_path': image_name, 'trial': trial,
                         'id': df_this_index_this_trial['id'], 'phase': phase}
            image_name = image_name.replace("/", "")
            save_dir = pathlib.Path(folder_name) / image_name
            save_dir.mkdir(parents=True, exist_ok=True)
            np.save(save_dir / "eye_cam.npy", info_dict)
            index_image += 1


file_phase = 'metadata_phase_3.csv'
create_heatmaps(
    eyetracking_dataset_path,
    file_phase,
    folder_name='./data/cxr_gaze/eye_cam',
    phase=1
>>>>>>> 3527c18f2d5fd8c863eb2645efb845bee913457a
)