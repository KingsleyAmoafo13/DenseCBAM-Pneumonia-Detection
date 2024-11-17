class Config:
    # Paths
    TRAIN_PATH = r"C:\Users\MarksLab\Downloads\archive\chest_xray\chest_xray\train"
    VALIDATION_PATH = r"C:\Users\MarksLab\Downloads\archive\chest_xray\chest_xray\val"
    TEST_PATH = r"C:\Users\MarksLab\Downloads\archive\chest_xray\chest_xray\test"
    LOG_DIR = "logs"
    MODEL_SAVE_PATH = "logs/best_model.pth"

    # Training Parameters
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5  # L2 Regularization
    EPOCHS = 20  # Adjust as needed

    # Seed for reproducibility
    SEED = 42
