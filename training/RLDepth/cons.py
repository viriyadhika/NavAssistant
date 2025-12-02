from torchvision import transforms
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ACTIONS = ["MoveAhead", "RotateRight", "RotateLeft"]
NUM_ACTIONS = len(ACTIONS)

IMG_SIZE = 224
GAMMA = 0.99
GAE_LAMBDA = 0.97
PPO_CLIP = 0.2
VALUE_COEF = 0.5
LR = 1e-4
TRAIN_EPOCHS = 20
MINIBATCHES = 4
ROLLOUT_STEPS = 2048
EPISODE_STEPS = ROLLOUT_STEPS // MINIBATCHES
MAX_GRAD_NORM = 0.5
FEAT_DIM = 128

transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),  # -> float in [0,1], shape (3,H,W)
        ])
