import argparse
from model_pipeline import prepare_data, train_model, save_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the model")
    args = parser.parse_args()

    if args.train:
        print("Preparing data...")
        X, y = prepare_data()
        
        print("Training model...")
        model = train_model(X, y)
        
        print("Saving model...")
        save_model(model)
