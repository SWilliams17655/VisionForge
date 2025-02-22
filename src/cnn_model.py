import torchvision
import torch.utils.data
import numpy as np
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
from tqdm import tqdm

#**********************************************************************************************************************

class RCNN_Model:
    def __init__(self, dataset = None, error_logger = None):
        """Region Based Convolution Neural Network (R-CNN) to detect objects in an image.
        @:param dataset -- Dataset to be used in training. If none, then does not train.
        @:param error_logger -- Logger to log errors during training. If none, bypasses all error logging.
        """

        self.error_logger = error_logger
        # Determines if GPU is available and sets device for loading data.
        if self.error_logger is not None:
            if torch.cuda.is_available():
                self.error_logger.info("This iteration will be using the GPU to accelerate processing; \n")
            else:
                self.error_logger.info(f"This iteration will NOT be using the GPU to accelerate processing.\n")
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Creates the dataset and data-loader.
        self.dataset = dataset

        if self.dataset is not None:
            train_size = int(0.8 * len(dataset))
            generator = torch.Generator().manual_seed(42) #Seeding the random generator to ensure consistent validation sets.
            train_set, val_set = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size], generator=generator)
            self.train_data_loader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True, collate_fn=collate_fn)
            self.val_data_loader = torch.utils.data.DataLoader(val_set, batch_size=3, shuffle=True, collate_fn=collate_fn)

        # Creates the initial model with
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)

        # Using pre-trained Faster R-CNN so replacing the classifier with a new model trained on this dataset.
        num_classes = 10
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # Moving the model to the GPU after replacing the head.
        self.model.to(self.device)

# **********************************************************************************************************************
    def load_model(self, filename):
        """Loads model from filename location.
        @:param filename -- Relative model.pkl location for use.
        """
        # Creates the initial model loading weights from a saved file.
        self.model = torch.load(filename, weights_only=False)
        self.model.eval()
        self.model.to(self.device)

# **********************************************************************************************************************
    def train_model(self, num_epochs):
        self.model.train()

        # Creating optimizer using SGD
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.5, weight_decay=1e-4)
        lr_scheduler = None
        loss_dict_val = ""
        for epoch in range(int(num_epochs)):
            for images, targets in tqdm(self.val_data_loader):  # Validating using validation data

                # Moves images and bounding boxes to GPU.
                images = list(image.to(self.device) for image in images)
                labels = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                try:
                    loss_dict_val = self.model(images, labels)  # Executes model with targets for validation.

                except torch.OutOfMemoryError:
                    self.error_logger.warning("Image exceeded memory capacity of GPU, the training algorithm will remove it from training.")

                except RuntimeError:
                    self.error_logger.error("Runtime Error")

                except AssertionError:
                    self.error_logger.warning(f"Bounding box error has been encountered. Area of the bounding box was < 0.")

                images = None  # Setting images and labels to None for garbage collection.
                labels = None
                torch.cuda.empty_cache()  # Empties the cache to minimize GPU memory overload.

            for images, labels in tqdm(self.train_data_loader):  # Training using training data.

                # Moves images and targets to the GPU.
                images = list(image.to(self.device) for image in images)
                labels = [{k: v.to(self.device) for k, v in t.items()} for t in labels]

                try:
                    loss_dict_train = self.model(images, labels)  # Executes model with targets and saves loss error results.

                    losses = sum(loss for loss in loss_dict_train.values())

                    loss_value = losses.item()

                    losses.backward()  # computes dloss/dx for every parameter.
                    optimizer.step()  # updates the value of x using the gradient x.gradient.

                except torch.OutOfMemoryError:
                    self.error_logger.warning("Image exceeded memory capacity of GPU, the training algorithm will remove it from training.")

                except RuntimeError:
                    self.error_logger.error("Runtime Error")

                except AssertionError:
                    self.error_logger.warning(f"Bounding box error")

                images = None  # Setting images and labels to None for garbage collection.
                labels = None
                optimizer.zero_grad()  # Clears x.gradient for every parameter X in the optimizer. Must be called before losses.backward.
                torch.cuda.empty_cache()  # Empties the cache to minimize GPU memory overload.

            print(loss_dict_val)
            torch.save(self.model, "Models/sat_scan_developmental.pth")

# **********************************************************************************************************************
    def process_image(self, image):

        self.model.eval()
        image = image.unsqueeze(0)
        # Moves images and bounding boxes to GPU.
        image = image.to(self.device)
        output = self.model(image)
        return output

# **********************************************************************************************************************
def collate_fn(batch):
    # Returns a collator for a Pytorch Dataloader.
    return tuple(zip(*batch))

# **********************************************************************************************************************
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    xi1 = max(x1, x1_p)
    yi1 = max(y1, y1_p)
    xi2 = min(x2, x2_p)
    yi2 = min(y2, y2_p)

    inter_area = max(0, xi2 - xi1 + 1) * max(0, yi2 - yi1 + 1)
    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x2_p - x2_p + 1) * (y2_p - y1_p + 1)

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area

def calculate_average_precision(recalls, precisions):
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])

    indices = np.where(recalls[1:] != recalls[:-1])[0]
    average_precision = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])

    return average_precision

def mean_average_precision(ground_truths, predictions, iou_threshold=0.5):
    average_precisions = []

    for class_id in set([gt[0] for gt in ground_truths]):
        gt_boxes = [gt[1] for gt in ground_truths if gt[0] == class_id]
        pred_boxes = [pred for pred in predictions if pred[0] == class_id]

        pred_boxes.sort(key=lambda x: x[2], reverse=True)

        true_positives = np.zeros(len(pred_boxes))
        false_positives = np.zeros(len(pred_boxes))
        detected_boxes = []

        for i, pred in enumerate(pred_boxes):
            pred_box = pred[1]
            best_iou = 0
            best_gt_idx = -1

            for j, gt_box in enumerate(gt_boxes):
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            if best_iou >= iou_threshold and best_gt_idx not in detected_boxes:
                true_positives[i] = 1
                detected_boxes.append(best_gt_idx)
            else:
                false_positives[i] = 1

        cumulative_true_positives = np.cumsum(true_positives)
        cumulative_false_positives = np.cumsum(false_positives)

        recalls = cumulative_true_positives / len(gt_boxes)
        precisions = cumulative_true_positives / (cumulative_true_positives + cumulative_false_positives)

        average_precision = calculate_average_precision(recalls, precisions)
        average_precisions.append(average_precision)

    return np.mean(average_precisions)