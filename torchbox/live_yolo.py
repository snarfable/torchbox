import cv2
from torch import device, load, cuda
from torchbox.yolo import YoloNet, infer_in_memory as infer
import argparse

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Configure options for live YOLO detection.")
    parser.add_argument("--cam-input", type=int, default=0)
    parser.add_argument("--model-path", type=str, default="models/yolo.pt")
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--classes", type=str, nargs='+', default=["Circle", "Rectangle", "Triangle"])
    return parser

def main():
    # Parse command-line arguments
    parser = build_arg_parser()
    args = parser.parse_args()

    # Set number of classes and grid size for YOLO model (adjust as needed)
    CLASSES = args.classes
    NUM_CLASSES = len(CLASSES)
    S = args.grid_size  # Grid size (S x S)

    # Set the model path (adjust to your trained model's path)
    model_path = args.model_path  # Replace with the actual path to your trained YOLO model

    # Set the device for inference (CPU or GPU)
    use_device = device("cuda" if cuda.is_available() else "cpu")

    # Set up the YOLO model for inference
    model = YoloNet(S, NUM_CLASSES).to(use_device) # Use CPU for inference; change to "cuda" if GPU is available
    state_dict = load(model_path, map_location=use_device)
    model.load_state_dict(state_dict)

    # 1. Initialize video source (0 for default webcam, or path to video file)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    # Calculate frame delay based on video FPS (or default to ~30 FPS / 33ms)
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps and fps > 0 else 30

    print("Starting video capture. Press 'q' to quit.")

    while True:
        # 2. Read frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Stream ended or failed to capture frame.")
            break

        # --- Frame processing logic goes here ---

        # 3. Update the window with the current frame
        frame_annotated = infer(model, use_device, frame)  # Assuming this function processes the frame and displays results
        cv2.imshow("Live Stream", frame_annotated)

        # 4. Wait for key press & handle window refresh ('q' to exit)
        if cv2.waitKey(delay) & 0xFF == ord("q"):
            break

    # 5. Clean up resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
