import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score

class EvaluationHelper:
    def __init__(self):
        self._qunantized_model_path = None
        self._test_generator = None
        self._tflite_interpretor = None

    @property
    def qunantized_model_path(self):
        return self._qunantized_model_path

    @qunantized_model_path.setter
    def qunantized_model_path(self, new_quantized_model_path):
        self._qunantized_model_path = new_quantized_model_path

    @property
    def test_generator(self):
        return self._test_generator

    @test_generator.setter
    def test_generator(self, new_generator):
        self._test_generator = new_generator

    @property
    def tflite_interpretor(self):
        self._tflite_interpretor = tf.lite.Interpreter(model_path=self.qunantized_model_path)
        return self._tflite_interpretor

    def get_model_predictions(self, model):
        """
        Use this method to  get prediction from .h5 models.
        """
        predictions = model.predict_generator(self.test_generator, steps=self.test_generator.n/self.test_generator.batch_size, verbose = True)
        return predictions

    def get_tflite_predictions(self):
        """
        Use this method to get predictions from tflite models.
        """
        print(type(self.test_generator.n))
        test_steps = self.test_generator.n / self.test_generator.batch_size
        prediction_array = np.empty((0, 14)) # Since NIH CXR has 14 classes.
        tflite_interpretor = self.tflite_interpretor
        tflite_interpretor.allocate_tensors()
        
        input_details = self.tflite_interpretor.get_input_details()[0]
        output_details = self.tflite_interpretor.get_output_details()[0]
        print("== Input details ==")
        print("shape:", input_details['shape'])
        print("type:", input_details['dtype'])
        print("\n== Output details ==")
        print("shape:", output_details['shape'])
        print("type:", output_details['dtype'])

        for idx in range(int(test_steps) + 1):
            image_batch, _ = next(self.test_generator)
            for item in image_batch:
                test_image = item
                test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
                tflite_interpretor.set_tensor(input_details['index'], test_image)
                tflite_interpretor.invoke()
                tflite_model_predictions = tflite_interpretor.get_tensor(output_details['index'])
                prediction_array = np.append(prediction_array, tflite_model_predictions, axis = 0)
            print(f"Processed : {idx}")

        return prediction_array

    def get_auc_roc_score(self, prediction_array):
        auc_score = roc_auc_score(self.test_generator.labels, prediction_array)
        return auc_score

    def get_auc_plot(self, prediction_array, all_labels, file_name):
        fig, c_ax = plt.subplots(1, 1, figsize = (9, 9))
        for (idx, c_label) in enumerate(all_labels):
            fpr, tpr, _ = roc_curve(self.test_generator.labels[:,idx].astype(int), prediction_array[:,idx])
            c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))
        
        c_ax.legend()
        c_ax.set_xlabel('False Positive Rate')
        c_ax.set_ylabel('True Positive Rate')
        fig.savefig(file_name)