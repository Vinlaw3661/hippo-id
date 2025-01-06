# 🦛 Hippo-ID

Hippo-ID is a zero-shot facial recognition engine designed for inference and identification. The name "Hippo-ID" is inspired by the hippocampus, the part of the brain responsible for memory, and its mascot is a friendly hippopotamus 🦛. Utilizing advanced tools and models, it can segment faces, identify known individuals, and interactively learn new identities using audio and text-to-speech technologies. Hippo-ID provides seamless database integration and supports real-time recognition through a webcam feed.

## ✨ Features

- **Zero-Shot Recognition**: Recognizes known individuals without retraining.
- **Interactive Learning**: Learns new faces by asking for names and acknowledging introductions.
- **Database Support**: Stores embeddings using ChromaDB or DeepFace.
- **Text-to-Speech and Speech-to-Text Integration**: Uses ElevenLabs and AssemblyAI for speech interactions.
- **Real-Time Processing**: Processes frames from a webcam stream with live feedback.

## 📋 Requirements

### Dependencies

- Python 3.9+
- Required Libraries:
  ```
  pip install numpy opencv-python mediapipe pyttsx3 pandas sounddevice SpeechRecognition assemblyai deepface langchain-core langchain-anthropic elevenlabs chromadb nanoid pydantic scipy python-dotenv tf-keras
  ```
- Supported Models and APIs:
  - ChromaDB
  - DeepFace
  - AssemblyAI
  - ElevenLabs
  - Anthropic Claude

### API Keys

Ensure the following API keys are set in a `.env` file:

```
ASSEMBLYAI_API_KEY=your_assemblyai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
ELEVENLABS_API_KEY=your_elevenlabs_api_key
```

## 🗂️ Directory Structure

```
Hippo-ID/
|-- utils/
|   |-- settings.py
|   |-- helpers.py
|-- hippo.py
|-- main.py
|-- .env
|-- outputs/
    |-- audio/
    |-- faces/
```

## ⚙️ Setup

1. Clone the repository and navigate to the project directory.
2. Install the required dependencies using the command:
   ```
   pip install -r requirements.txt
   ```
3. Set up the `.env` file with your API keys.
4. Create necessary directories for outputs:
   ```
   mkdir -p outputs/audio outputs/faces
   ```

## 🚀 Usage

### Running the Application

### Example Code

Below is a simple code snippet to demonstrate how the Hippo-ID engine works:

```python
from hippo import Hippo
from utils.helpers import StorageMode
import cv2

# Initialize the Hippo-ID engine
hippo = Hippo(database_path="chroma", storage_mode=StorageMode.CHROMA, use_elevenlabs=True, use_assemblyai=True)

# Load an image for identification
image_path = "path_to_image.jpg"
image = cv2.imread(image_path)

# Perform identification
result, message = hippo.identify(image, verbose=True)

if result:
    print(f"New person identified: {message}")
else:
    print(f"Person already known: {message}")
```

Run the `main.py` file to start the real-time facial recognition system:

```bash
python main.py
```

### Interactions

1. **Known Face**: When a known face is detected, the system acknowledges the individual.
2. **Unknown Face**: If an unknown face is detected:
   - The system segments the face.
   - Asks for the person's name using text-to-speech.
   - Listens to the response and stores the identity in the selected storage mode (ChromaDB or DeepFace).

### Storage Modes

- **ChromaDB**: Stores face embeddings for fast retrieval.
- **DeepFace**: Uses a directory-based database to store faces.

## 🛠️ Code Overview

### `utils/settings.py`

- Handles environment variables and initializes:
  - ChromaDB client.
  - AssemblyAI transcriber.
  - ElevenLabs voice model.
  - Anthropic Claude LLM.

### `utils/helpers.py`

- Provides utility functions:
  - Face embedding creation and storage.
  - Face segmentation using MediaPipe.
  - Voice interaction (recording, transcribing, TTS).
  - LLM-based question formulation.

### `hippo.py`

- Defines the `Hippo` class for:
  - Identifying faces.
  - Storing and retrieving embeddings.
  - Asking for names and acknowledging individuals.

### `main.py`

- Runs the real-time webcam feed for face recognition using the `Hippo` engine.

## 🎯 Example Output

### Output Explanation

- **(False, Valid Name)**: The person is already known, so no identification was required.
- **(True, Valid Name)**: A new face was identified, and the name was successfully determined.
- **(False, FaceState.UNDETECTED)**: The system could not detect a face in the segmented image.

### Examples

- **Known Face**:
  ```
  (False, 'John Doe')
  ```
- **Unknown Face**:
  ```
  (True, 'Jane Smith')
  ```

## 🚧 Future Enhancements

- Add support for additional face recognition models.
- Enhance real-time performance with GPU acceleration.
- Integrate multi-person detection and recognition.

## 📜 License

This project is licensed under the MIT License.

---



