import torch
from configs import ConfigBase
from evaluator import ProbingEvaluator, ProbingConfig
from dataset import create_wall_dataloader

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

    # Create DataLoaders instead of raw datasets
    probe_train_ds = create_wall_dataloader(
        data_path=probe_train_path,
        probing=True,
        device='cpu',
        batch_size=config.batch_size,
        train=True,
        #augment=False
    )

    probe_normal_val_ds = create_wall_dataloader(
        data_path=probe_normal_val_path,
        probing=True,
        device='cpu',
        batch_size=config.batch_size,
        train=False,
        #augment=False
    )

    probe_wall_val_ds = create_wall_dataloader(
        data_path=probe_wall_val_path,
        probing=True,
        device='cpu',
        batch_size=config.batch_size,
        train=False,
        #augment=False
    )

    probe_val_ds = {
        'normal': probe_normal_val_ds,
        'wall': probe_wall_val_ds,
    }

    return probe_train_ds, probe_val_ds

def load_model(config):
    """Load or initialize the model."""
    from models import JEPA_Model
    model = JEPA_Model(
        repr_dim=config.repr_dim,
        action_dim=config.action_dim,
        device=config.device
    ).to(config.device)
    model.load_state_dict(torch.load(config.model_weights_path, map_location=config.device))
    model.eval()
    return model

def evaluate_model():
    config = MainConfig()

    probe_train_ds, probe_val_ds = load_data(config)
    model = load_model(config)

    evaluator = ProbingEvaluator(
        device=config.device,
        model=model,
        probe_train_ds=probe_train_ds,  # DataLoader now
        probe_val_ds=probe_val_ds,      # Dict of DataLoaders now
        config=ProbingConfig(),
        quick_debug=False,
    )

    prober = evaluator.train_pred_prober()
    avg_losses = evaluator.evaluate_all(prober=prober)

    for dataset_name, loss in avg_losses.items():
        print(f'Average evaluation loss on {dataset_name} dataset: {loss:.4f}')

if __name__ == '__main__':
    evaluate_model()
