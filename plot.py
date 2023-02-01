import myanfis
import pandas as pd
import tensorflow as tf
import myanfis
from keras.models import load_model
import numpy as np

base_model = load_model("C:\\Users\\Hari\\PycharmProjects\\pythonProject\\prediction.h5", custom_objects={'FuzzyLayer': myanfis.FuzzyLayer, 'RuleLayer':myanfis.RuleLayer, 'NormLayer':myanfis.NormLayer, 'DefuzzLayer':myanfis.DefuzzLayer, 'SummationLayer':myanfis.SummationLayer})


def prediction(test_data):
    pred = base_model.predict(test_data)
    print(pred)

test_data = np.array([30])
prediction(test_data)