import torch
from torch.utils.data import DataLoader
from configs import ConfigBase
from evaluator import ProbingEvaluator, ProbingConfig
from dataset import WallDataset


class MainConfig(ConfigBase):
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size: int = 64
    num_workers: int = 4
    repr_dim: int = 256
    action_dim: int = 2
    model_weights_path: str = 'model_weights.pth'


def load_data(config):
    # Paths to the datasets
    probe_train_path = '/scratch/DL24FA/probe_normal/train/'
    probe_normal_val_path = '/scratch/DL24FA/probe_normal/val/'
    probe_wall_val_path = '/scratch/DL24FA/probe_wall/val/'

    # Create datasets
    probe_train_ds = WallDataset(probe_train_path, batch_size=config.batch_size)
    probe_normal_val_ds = WallDataset(probe_normal_val_path, batch_size=config.batch_size)
    probe_wall_val_ds = WallDataset(probe_wall_val_path, batch_size=config.batch_size)

    # Dictionary of validation datasets
    probe_val_ds = {
        'normal': probe_normal_val_ds,
        'wall': probe_wall_val_ds,
    }

    return probe_train_ds, probe_val_ds


def load_model(config):
    """Load or initialize the model."""
    ################################################################################
    # TODO: Initialize and load your model
    from models import JEPA_Model

    model = JEPA_Model(
        repr_dim=config.repr_dim,
        action_dim=config.action_dim,
        device=config.device
    ).to(config.device)

    # Load trained weights
    model_weights_path = config.model_weights_path
    model.load_state_dict(torch.load(model_weights_path, map_location=config.device))

    model.eval()  # Set model to evaluation mode
    ################################################################################
    return model


def evaluate_model():
    config = MainConfig()

    # Load data
    probe_train_ds, probe_val_ds = load_data(config)

    # Load model
    model = load_model(config)

    # Initialize the evaluator
    evaluator = ProbingEvaluator(
        device=config.device,
        model=model,
        probe_train_ds=probe_train_ds,
        probe_val_ds=probe_val_ds,
        config=ProbingConfig(),
        quick_debug=False,
    )

    # Train the prober
    prober = evaluator.train_pred_prober()

    # Evaluate the model
    avg_losses = evaluator.evaluate_all(prober=prober)

    # Print evaluation results
    for dataset_name, loss in avg_losses.items():
        print(f'Average evaluation loss on {dataset_name} dataset: {loss:.4f}')


if __name__ == '__main__':
    evaluate_model()
