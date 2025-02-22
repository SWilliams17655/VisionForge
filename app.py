from flask import Flask, render_template
from src.cnn_model import RCNN_Model
import numpy as np
import torch
import torchvision
import cv2
import logging
from torchvision.transforms import v2
from torchvision.tv_tensors import BoundingBoxes
from tqdm import tqdm
import torch.utils.data
import glob
import folium
import rasterio
from src.sat_image_dataset import SatImageDataset as image_dataset

app = Flask(__name__)

data = []
m = folium.Map(tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
               attr='My Data Attribution')
m.get_root().width = "3000px"
m.get_root().height = "600px"

#**********************************************************************************************************************
def evaluate_image(image_array, model):
    """
    Uses computer vision algorithm to process an image detecting maritime vessels.
    :param image_array: A 3d numpy array representing the image to be processed.
    :param model: The model to be used.
    :return: Tuple including image with boxes drawn and raw bounding boxes.
    """

    #The transform used to compose the image for processing.
    transform = v2.Compose([
        v2.ToDtype(torch.float16, scale=True),
        v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    ])

    # Reads image then converts to tensor per documentation
    # https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html#defining-the-dataset.
    image = transform(image_array)
    image_tensor = torchvision.tv_tensors.Image(image)

    # Detects maritime vessels in image.
    bboxes = model.process_image(image_tensor)

    # Uses non-maximum suppression to remove overlapping boxes.
    detections = torchvision.ops.nms(bboxes[0]['boxes'], bboxes[0]['scores'], .2)

    # Converts the image to an array to draw bounding boxes then loops to draw bounding boxes.
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

    print(bboxes)

    for i in detections:
        if bboxes[0]['scores'][i] > .05:
            cv2.rectangle(image_array, (int(bboxes[0]['boxes'][i][0]),
                                        int(bboxes[0]['boxes'][i][1])),
                                       (int(bboxes[0]['boxes'][i][2]),
                                        int(bboxes[0]['boxes'][i][3])),
                                       (0, 255, 0),
                                       3)

    return image_array, bboxes

#**********************************************************************************************************************
@app.route("/", methods=['GET', 'POST'])
def home_page():
    return render_template("index.html", map=m._repr_html_())

#**********************************************************************************************************************

logger = logging.getLogger("SAT-SCAN Log")
fileout = logging.FileHandler('logs/SAT-SCAN.log', mode='w')
fileout.setLevel(logging.INFO)
logger.setLevel(logging.INFO)
logger.addHandler(fileout)
fmt = logging.Formatter("%(name)s: %(asctime)s | %(levelname)s | %(filename)s:%(lineno)s | %(process)d >>> %(message)s")
fileout.setFormatter(fmt)

if __name__ == "__main__":
    app.run(host="0.0.0.0")

ans = input("\nWhat would you like to do: \n [1] Train new model. \n [2] Evaluate image. \n [3] Exit\n")

if ans == "1":
    training_data = image_dataset('training data/training.json', logger)
    model = RCNN_Model(training_data, logger)
    epoches = 2

    model.train_model(epoches)

if ans == "2":

    model = RCNN_Model(None, logger)

    model.load_model("Models/marsatscan.pth") # Load trained model.
    images = []

    for file in tqdm(glob.glob("training data/val_images/*.tif")):

        # Loads satellite image then reshapes to standard used in pyvision
        d = rasterio.open(file)
        image = d.read()
        image = np.transpose(image, (1, 2, 0))
        image_array, bboxes = evaluate_image(image, model)

        # Adds image to the folium map by embedding image at geographic location.
        folium.Marker([d.bounds.top, d.bounds.left]).add_to(m)
        folium.raster_layers.ImageOverlay(
            image=cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB),
            bounds=[[d.bounds.bottom, d.bounds.left], [d.bounds.top, d.bounds.right]]).add_to(m)

