import warnings
warnings.filterwarnings("ignore")
from experiment_manager import ExperimentManager
from dataset import VisiumHDDataset
from model import VisiumHDFPNModel

if __name__ == "__main__":
    manager = ExperimentManager()
    opt = manager.get_opt()

    print("Initializing pred dataset")
    pred_dataset = VisiumHDDataset(manager)
    pred_dataset.set_train(False)

    gene_map_shape = pred_dataset.gene_map.shape

    model = VisiumHDFPNModel(manager, gene_map_shape)
    print("Loading model from ", opt.load_path)
    model.load(opt.load_path)
    model.predict(pred_dataset)