import warnings
warnings.filterwarnings("ignore")
from experiment_manager import ExperimentManager
from dataset import VisiumHDDataset
from model import VisiumHDFPNModel


if __name__ == "__main__":
    manager = ExperimentManager()
    opt = manager.get_opt()

    # Initializing the Dataset
    print("Initializing train dataset")
    train_dataset = VisiumHDDataset(manager)
    train_dataset.set_train(True)
    gene_map_shape = train_dataset.gene_map.shape

    # Initializing the model
    model = VisiumHDFPNModel(manager, gene_map_shape)
    model.train_model(train_dataset)

    print("Initializing pred dataset")
    pred_dataset = VisiumHDDataset(manager)
    pred_dataset.set_train(False)
    model.predict(pred_dataset)
