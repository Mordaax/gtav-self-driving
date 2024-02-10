# train_model.py

import numpy as np
from models import alexnet2
WIDTH = 160
HEIGHT = 120
LR = 1e-3
EPOCHS = 10
MODEL_NAME = './models/pygta5-car-fast-{}-{}-{}-epochs-300K-data.model'.format(LR, 'alexnetv2.2',EPOCHS)

model = alexnet2(WIDTH, HEIGHT, LR)

""" train_data = np.load('./data/training_data_balanced.npy', allow_pickle=True)

train = train_data[:-100]
test = train_data[-100:]

X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=10, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

model.save(MODEL_NAME) """


hm_data = 16
for i in range(EPOCHS):
    for i in range(1,hm_data+1):
        train_data = np.load('./data/training_data-{}.npy'.format(i), allow_pickle=True)

        train = train_data[:-100]
        test = train_data[-100:]

        X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,1)
        Y = [i[1] for i in train]

        test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1)
        test_y = [i[1] for i in test]

        model.fit({'input': X}, {'targets': Y}, n_epoch=1, validation_set=({'input': test_x}, {'targets': test_y}), 
            snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

        model.save(MODEL_NAME)



# tensorboard --logdir=foo:D:\pygta5\MyProject\log\





