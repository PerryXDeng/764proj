import numpy as np
import tensorflow.compat.v1 as tf
import os
import cv2
from model import cnn_model_fn


tf.logging.set_verbosity(tf.logging.INFO)



def load(dimension):

    imagesTop = []
    imagesSide = []
    imagesFront = []

    ls = 0
    folder = "/localhome/ama240/Desktop/LeChairs/evaluate-chairs/"

    length = len(os.listdir(folder)) // 3
    ls += length

    for filename in os.listdir(folder):

        view = int(filename.split(".")[0])
        view = view % 3

        img = cv2.imread(folder+filename)
        if dimension < 224:
            img = cv2.resize(img, dsize=(dimension, dimension), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.nan_to_num(img)

        #This relies on the files being loaded in order. For that to happen, the 0 padding in the file name is crucial.
        #If you do not have that, then you need to change the logic of this loop.

        if img is not None:
            if view == 2:
                imagesSide.append(1. - img / 255.)
            elif view == 0:
                imagesTop.append(1. - img / 255.)
            else:
                imagesFront.append(1. - img / 255.)

    imagesTop = np.array(imagesTop)
    imagesFront = np.array(imagesFront)
    imagesSide = np.array(imagesSide)

    #flatten the images
    imagesTop = np.reshape(imagesTop, (ls, dimension * dimension))
    imagesFront = np.reshape(imagesFront, (ls, dimension * dimension))
    imagesSide = np.reshape(imagesSide, (ls, dimension * dimension))

    return imagesTop, imagesFront, imagesSide

def main(*argv):

    #load chairs dataset
    imagesTop, imagesFront, imagesSide = load(56)

    test_images = {}
    #then the rest are test images and labels
    test_images["Top"] = imagesTop
    test_images["Front"] = imagesFront
    test_images["Side"] = imagesSide

    test_evaluations = [[],[],[]]

    for id, view in enumerate(["Front", "Side", "Top"]):
        classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="/localhome/ama240/Desktop/LeChairs/checkpoint/"+view+"/")

        eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": test_images[view]},
                                                           num_epochs=1,
                                                           shuffle=False)

        #The line below returns a generator that has the probability that the tested samples are Positive cases or Negative cases
        eval_results = classifier.predict(input_fn=eval_input_fn)

        #You need to iterate over the generator returned above to display the actual probabilities
        #This line should print something like {'classes': 0, 'probabilities': array([0.91087427, 0.08912573])}
        #the first element of 'probabilities' is the correlation of input with the Negative samples. The second element means positive.
        #If you evaluate multiple samples, just keep iterating over the eval_results generator.
        #eval_instance = next(eval_results)
        #print(eval_instance)

        # This is how you extract the correlation to the positive class of the first element in your evaluation folder
        for eval in eval_results:
            #print("probability that this instance is positive is %3.2f " % eval['probabilities'][1])
            test_evaluations[id].append(eval['probabilities'][1])

    #the probability that the chair is a positive example is given by the minimum of the probabilities from each of the three views
    #in the default configuration sent, the first ten chairs should be negatives (low value) and the ten last chairs should be positives (high value)
    #as can be seen in this quick evaluation, there is roon for inprovement in the algorithm

    print("The probability that the chair is a positive example is given by the minimum of the probabilities from each of the three views.")
    print("In the default configuration sent, the first ten chairs should be negatives (low value) and the ten last chairs should be positives (high value).")
    print("There is a lot of room for improvement here!")
    evaluation_chairs = np.amin(test_evaluations, axis=0)
    print(evaluation_chairs)


if __name__ == "__main__":
    # Add ops to save and restore all the variables.
    #saver = tf.train.Saver()
    tf.app.run()