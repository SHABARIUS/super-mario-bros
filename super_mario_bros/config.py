"""Main configuration file."""

from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# Training location and model artifacts
TRAINING_LOCATION = "gcp"
BUCKET_NAME = "super-mario-bros"
ARTIFACT_FOLDER = None
SAVE_PATH = "train"
LOG_PATH = "logs"
TRAINING_SUMMARY = True
NAME_PREFIX = "smb"

# Model and training parameters
N_ENVS = 8
ENV_KWARGS = None
JOYPAD_SPACE = SIMPLE_MOVEMENT
NUM_STACK = 4
IMAGE_RESIZE = 64
SKIP = 4
POLICY = "CnnPolicy"
SAVE_FREQ = 500000
EVAL_FREQ = 50000
EVAL_LVL = "SuperMarioBros-1-1-v0"
TOTAL_TIMESTEPS = 5000000
N_STEPS = 4096
BATCH_SIZE = 64
N_EPOCHS = 10
POLICY_KWARGS = None
LEARNING_RATE = 0.0003
GAMMA = 0.99
VERBOSE = 0
