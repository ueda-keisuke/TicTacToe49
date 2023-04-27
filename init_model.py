from keras import Input
from keras.models import Model
from funcy import *
from keras.saving.saving_api import save_model

from train import computational_graph

model = Model(*juxt(identity, computational_graph())(Input(shape=(3, 3, 2))))
save_model(model, 'model/candidates/0000.h5')
