import cv2
import time 
from hippo import Hippo
from utils.helpers import StorageMode, FaceState


hippo = Hippo(database_path="chroma", storage_mode=StorageMode.CHROMA, use_elevenlabs=True, use_assemblyai=True, audio_path="audio.wav", audio_save_directory="./outputs/audio")



def video_stream():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        
        return 
    
    start_time = time.time()
    frame_count = 0
    
    while True:

        ret, frame = cap.read()

        if not ret:
            print("Could not read frame")
            break

        frame_count += 1
        elapsed_time = time.time() - start_time
        
        if elapsed_time > 0:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()

        else:
            fps = 0

        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Webcam", frame)
        result, message = hippo.identify(frame, verbose=True)
        print((result, message))

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    video_stream()

if __name__ == "__main__":
    main()


