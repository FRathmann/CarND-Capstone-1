from styx_msgs.msg import TrafficLight
import numpy as np
import tensorflow as tf
import time
import rospy

'''
Source codes:
https://github.com/googlecodelabs/tensorflow-for-poets-2/blob/master/scripts/label_image.py
'''

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.model_file = rospy.get_param('~model') #'light_classification/retrained_graph.pb'
        self.graph = self.load_graph(self.model_file)
        self.labels = ['green', 'none', 'red', 'yellow']
        self.light_states = [TrafficLight.GREEN, TrafficLight.UNKNOWN, TrafficLight.RED, TrafficLight.YELLOW]
        self.sess = tf.Session(graph=self.graph)

    def load_graph(self, model_file):
        graph = tf.Graph()
        graph_def = tf.GraphDef()

        with open(model_file, "rb") as f:
            graph_def.ParseFromString(f.read())
        with graph.as_default():
            tf.import_graph_def(graph_def)
        return graph

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        input_height=224
        input_width=224
        input_mean=0
        input_std=255

        input_operation = self.graph.get_operation_by_name('import/input');
        output_operation = self.graph.get_operation_by_name('import/final_result');

        with self.graph.as_default():
            start = time.time()
            dims_expander = tf.expand_dims(image, 0);
            resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
            normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
            tensor = self.sess.run(normalized)
            results = self.sess.run(output_operation.outputs[0], {input_operation.outputs[0]: tensor})
            end=time.time()
            rospy.logwarn('\nEvaluation time (1-image): {:.3f}s\n'.format(end-start))

        results = np.squeeze(results)
        top_k = results.argsort()[-5:][::-1]
        label_idx = top_k[0]
        template = "{} (score={:0.5f})"
        for i in top_k:
            rospy.logwarn(template.format(self.labels[i], results[i]))
        return self.light_states[label_idx]
