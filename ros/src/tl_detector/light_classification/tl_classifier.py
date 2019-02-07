#!/usr/bin/env python
import time

import numpy as np
import rospy
import tensorflow as tf
from styx_msgs.msg import TrafficLight


class TLClassifier(object):
    def __init__(self):

        PATH_TO_GRAPH = r'frozen_inference_graph.pb'

        self.graph = tf.Graph()
        self.threshold = .7
        self.direct_hit_threshold = .95

        with self.graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_GRAPH, 'rb') as fid:
                od_graph_def.ParseFromString(fid.read())
                tf.import_graph_def(od_graph_def, name='')

            self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
            self.boxes = self.graph.get_tensor_by_name('detection_boxes:0')
            self.scores = self.graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.graph.get_tensor_by_name('num_detections:0')

        self.sess = tf.Session(graph=self.graph)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        with self.graph.as_default():
            img_expand = np.expand_dims(image, axis=0)
            start = time.time()
            (boxes, scores, classes, num_detections) = self.sess.run(
                [self.boxes, self.scores, self.classes, self.num_detections],
                feed_dict={self.image_tensor: img_expand})
            # rospy.loginfo("Prediction took {0:.2}s".format(time.time() - start))

        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        good_classes = list(classes[scores > self.threshold])
        if len(good_classes) == 0:
            return TrafficLight.UNKNOWN

        most_common_class = good_classes[0]
        if len(good_classes) > 1:
            if scores[0] < self.direct_hit_threshold:
                # fastest way to find most common item for small list according to https://stackoverflow.com/a/28528632/9309705
                most_common_class = max(map(lambda val: (good_classes.count(val), val), set(good_classes)))[1]

        if most_common_class == 1:
            return TrafficLight.GREEN
        elif most_common_class == 2:
            return TrafficLight.RED
        elif most_common_class == 3:
            return TrafficLight.YELLOW
