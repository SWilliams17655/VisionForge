import base64
from dataset import MNIST_Dataset
from flask import Flask, render_template, request
from io import BytesIO
from matplotlib.figure import Figure
import numpy as np
import neural_network as nn
import tensorflow as tf

app = Flask(__name__)

global y_pred_nn

train_data = MNIST_Dataset("Data/train.csv")
test_data = MNIST_Dataset("Data/test.csv")

train_data.load()
train_data.normalize()

test_data.load()
test_data.normalize()

ans = input("Would you like to train a new neural network [1] or use the last trained network [2]? ")

if ans == str(1):
    model_nn_clf = nn.train_neural_network(train_data.x, train_data.y, test_data.x, test_data.y)
    model_nn_clf.save("model.h5")

elif ans == str(2):
    model_nn_clf = tf.keras.models.load_model("model.h5")
else:
    print("Invalid Option: Please select 1 or 2.")

y_pred_nn = model_nn_clf.predict(test_data.x)


@app.route("/")
def home_page():
    data = []
    return render_template("MNIST_search_engine.html", data=data)


@app.route("/search", methods=["GET", "POST"])
def search():
    global y_pred_nn
    data = []

    for i in range(700):
        if np.argmax(y_pred_nn[i]) == int(request.form["selector"]):
            probability = y_pred_nn[i][np.argmax(y_pred_nn[i])]
            if probability > .99:
                print("/n")
                print(y_pred_nn[i])
                print(np.argmax(y_pred_nn[i]))
                print(probability)

                image = test_data.get_image_at(i)
                fig = Figure()
                ax = fig.subplots()
                ax.imshow(image, cmap='gray')

                # Save image to a temporary buffer.
                buf = BytesIO()
                fig.savefig(buf, format="png")

                data.append(base64.b64encode(buf.getbuffer()).decode("ascii"))

    return render_template("MNIST_search_engine.html", data=data)


if __name__ == "__main__":
    app.run(host="0.0.0.0")
