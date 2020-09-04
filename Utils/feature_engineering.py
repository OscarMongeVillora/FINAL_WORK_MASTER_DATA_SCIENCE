
import numpy as np

def stack_features(name_dataset, size_dat, SIZE_TRAIN, x_train_week, x_test_week, feature_wave = 1, feature_sign = 1):

    if name_dataset == 'casos_update':

        if feature_wave == 1:

            vec_first_wave = list(np.tile(0, 120))
            vec_second_wave = list(np.tile(1, size_dat - len(vec_first_wave)))
            vec_first_wave.extend(vec_second_wave)
            vec_waves = vec_first_wave

            vec_wave_train = np.array(vec_waves[:SIZE_TRAIN])
            vec_wave_test = np.array(vec_waves[SIZE_TRAIN:])

            x_train_week_wave = np.hstack((x_train_week, vec_wave_train.reshape(-1,1)))
            x_test_week_wave = np.hstack((x_test_week, vec_wave_test.reshape(-1, 1)))
            x_train_week = x_train_week_wave
            x_test_week = x_test_week_wave

        if feature_sign == 1:
            # [0-14: 0] [14-49 : 1] [49-100 : -1] [100-134: 0] [134-170: 1]
            vec_sign = list(np.tile(0, 15))
            vec_sign.extend(np.tile(1, 35)), vec_sign.extend(np.tile(-1, 51))
            vec_sign.extend(np.tile(0, 34)), vec_sign.extend(np.tile(1, 36))

            vec_sign_train = np.array(vec_sign[:SIZE_TRAIN])
            vec_sign_test = np.array(vec_sign[SIZE_TRAIN:])

            x_train_week_sign = np.hstack((x_train_week, vec_sign_train.reshape(-1,1)))
            x_test_week_sign = np.hstack((x_test_week, vec_sign_test.reshape(-1, 1)))
            x_train_week = x_train_week_sign
            x_test_week = x_test_week_sign

    #In case the name of the dataset is not one of the previous one, we return the input for x_train_week and x_test_week
    return [x_train_week, x_test_week]
