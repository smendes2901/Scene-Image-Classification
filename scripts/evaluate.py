import argparse
from models.model_utils import load_model
from data.data_preprocessing import get_test_loader


def evaluate(model_name):
    test_loader = get_test_loader()
    model = load_model(model_name)
    model.evaluate(test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the image classification model."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Specify model: resnet101 or se_resnet50",
    )
    args = parser.parse_args()
    evaluate(args.model)
