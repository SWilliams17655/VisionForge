import pandas as pd

class TrainingLossLog:

    def __init__(self):
        """ This class saves history of batch training to provide an average accuracy across batch and epochs. It is
        they exported to an .csv for use with second party data visualization software like PowerBi or excel."""

        self.__batch_history = pd.DataFrame([], columns=['ROI_loss', 'classification_loss'])
        self.__epoch_history = pd.DataFrame([], columns=['ROI_loss_avg', 'classification_loss_avg'])
        self.__epoch_start_marker = 0

#**********************************************************************************************************************

    def __getattr__(self, item):
        """Returns the item properly formatted"""

        match item:
             case 'log':
                 return self.__epoch_history

             case 'average_ROI_loss':
                if len(self.__batch_history) > 1:
                   return self.__batch_history.loc[self.__epoch_start_marker:]['ROI_loss'].mean()
                else:
                   return 0

             case 'average_classification_loss':
                if len(self.__batch_history) > 1:
                    return self.__batch_history.loc[self.__epoch_start_marker:]['classification_loss'].mean()
                else:
                    return 0
        return None

# *********************************************************************************************************************

    def send_batch(self, roi_loss, classification_loss):
        """Receives roi loss and classification loss then adds them to the history displayed.

        @:param roi_loss: ROI loss value
        @:param classification_loss: classification loss value
        """

        new_row = {'ROI_loss': float(roi_loss), 'classification_loss': float(classification_loss)}
        self.__batch_history.loc[len(self.__batch_history)] = new_row

# *********************************************************************************************************************

    def new_epoch(self):
        """Creates a new epoch in the batch history."""

        new_row = {'ROI_loss_avg': float(self.__batch_history.loc[self.__epoch_start_marker:]['ROI_loss'].mean()),
                   'classification_loss_avg': float(self.__batch_history.loc[self.__epoch_start_marker:]['classification_loss'].mean())}
        self.__epoch_history.loc[len(self.__epoch_history)] = new_row
        self.__epoch_start_marker = len(self.__batch_history) - 1
        print(self.__epoch_history)
        return new_row

# *********************************************************************************************************************

    def save_to_csv(self, epoch_filename):
        """Saves the history of training to batch_history.csv and epoch_history.csv file."""

        self.__epoch_history.to_csv(epoch_filename)