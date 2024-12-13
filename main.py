import cv2
from embeddings.extract_embeddings import extract_embeddings
from recognition.output_processing import process_videos
from config import CONFIG

def play_video(video_path):
    """Play a video file in a window."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return

    print("Playing processed video...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video playback complete.")
            break

        cv2.imshow('Processed Video Playback', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting video playback...")
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Step 1: Extract embeddings
    print("Extracting embeddings...")
    extract_embeddings(CONFIG["dataset_dir"], CONFIG["embedding_pt_dir"], CONFIG["device"])

    # Step 2: Process videos and save outputs
    print("Processing videos...")
    process_videos(CONFIG["input_video_dir"], CONFIG["output_dir"], CONFIG["embedding_pt_dir"], CONFIG["device"], CONFIG["confidence_threshold"])

    # Step 3: Automatically play one of the processed videos
    # Replace "example_output_video.mp4" with a path to one of your processed videos
    processed_video_path = f"{CONFIG['output_dir']}/example_output_video.mp4"
    play_video(processed_video_path)
