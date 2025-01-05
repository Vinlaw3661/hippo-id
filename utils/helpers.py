import numpy as np 
import chromadb
from nanoid import generate
from enum  import Enum
import os 
import cv2
import time
import base64
import mediapipe as mp
import pyttsx3
import pandas as pd
import sounddevice as sd
import speech_recognition as sr
import assemblyai as aai
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from deepface import DeepFace
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from elevenlabs.client import ElevenLabs
from elevenlabs import stream, VoiceSettings
from scipy.io.wavfile import write
from utils.helpers import record_audio

    
# Load environment variables and set up API keys
load_dotenv()
assemblyai_api_key = os.getenv("ASSEMBLYAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")


# Set up ChromaDB for persisting embeddings
chromadb_client = chromadb.PersistentClient()
collection = chromadb_client.get_or_create_collection("Faces")

# Set up LLM
llm  = ChatAnthropic(
    model = "claude-3-5-sonnet-20240620",
    temperature = 0.5,
    max_tokens = 1024,
    timeout = None,
    max_retries = 2,
    api_key= anthropic_api_key
)

# Set up text to speech model
voice = ElevenLabs(
    api_key = elevenlabs_api_key
)

# Set up speech to text model
aai.settings.api_key = assemblyai_api_key
transcriber = aai.Transcriber()


class PersonName(BaseModel):
    """Name of person extracted from provided text"""
    name: str = Field(description="The name of the person")

class FaceState(Enum):
    UNKNOWN = "Unknown"
    UNDETECTED = "Undetected"

class StorageMode(Enum):
    CHROMA = "Chroma"
    DEEPFACE = "DeepFace"

#--------------------------UTILITY-FUNCTIONS--------------------------

# ChoromaDB embedding functions
def create_face_embedding(img_path: str) -> np.ndarray:
    return DeepFace.represent(img_path=img_path, model_name='Facenet')[0]["embedding"]

def store_face_embedding(img_path: str, name: str) -> bool:

    embedding = create_face_embedding(img_path)

    if embedding is None:
        return False
    
    uid = generate(size=7)

    collection.add(
        embeddings=embedding,
        metadatas=[{"name": name}],
        ids=[uid]
    )

    return True


# Audio recording function
def record_audio(duration: int = 5, file_name: str = "audio.wav", save_directory: str = "./outputs/audio") -> str:
    fs = 44100

    print(f"Recording audio for {duration} seconds...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=2, dtype='int16')
    sd.wait()
    print("Audio recording complete.")

    os.makedirs(save_directory, exist_ok=True)

    audio_path = os.path.join(save_directory, file_name)
    write(audio_path, fs, audio)

    print(f"Audio saved to {audio_path}")
    
    return audio_path

# Face segmentation function
def segment_faces(img: np.ndarray, image_save_path: str = "./outputs/faces") -> tuple[np.ndarray,str, bool]:
    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

 
    height, width, _ = img.shape

    # Detect faces
    results = face_detection.process(img)

    # Check if any face was detected
    if results.detections:
        # Process the first detected face
        detection = results.detections[0]
        bboxC = detection.location_data.relative_bounding_box

        # Convert bounding box to pixel values
        x = int(bboxC.xmin * width)
        y = int(bboxC.ymin * height)
        w = int(bboxC.width * width)
        h = int(bboxC.height * height)

        # Ensure coordinates are within image bounds
        x = max(0, x)
        y = max(0, y)
        w = min(width - x, w)
        h = min(height - y, h)

        # Crop the face using the bounding box
        face_roi = img[y:y+h, x:x+w]
        ouput_path = f"{image_save_path}/masked_image.png"
        os.makedirs(os.path.dirname(ouput_path), exist_ok=True)

        return face_roi, ouput_path, True
    else:

        return np.array([]), FaceState.UNDETECTED, False




#--------------------------FACE RECOGNITION--------------------------

# Function to check if a person is already known using Chroma
def is_known_face(face_path:str) -> tuple[bool, str]:
    embedding = create_face_embedding(face_path)

    if embedding is None:
        return False, FaceState.UNDETECTED

    matches =  collection.query(
                    query_embeddings=embedding,
                    n_results=1,
                )
    
    if len(matches["ids"]) == 0:
        return False, FaceState.UNKNOWN
    
    distance = matches["distances"][0][0]

    if distance < 0.55:
        return True, matches["metadatas"][0][0]["name"]
    
# Function to check if a person is already known using DeepFace
def is_known_face_deepface(face_path: str, database_path: str) -> tuple[bool, str]:

    faces = os.listdir(database_path)
    if len(faces) == 0:
        return False, FaceState.UNKNOWN
    
    df = DeepFace.find(
        img_path=face_path,
        db_path=database_path,
        model_name='Facenet'
    )
    df = df[0]
    if df.shape[0] > 0:
        name = df.iloc[0]["identity"].split('\\')[-1].split(".")[0].title()
        return True, name
    else:
        return False, FaceState.UNDETECTED
    


#--------------------------PERSON DESCRIPTION & NAME ASKING--------------------------

# Function to describe a person if they are not known
def describe_person(img_path:str) -> str:

    if not os.path.exists(img_path):
        raise FileNotFoundError("Image not found")

    with open(img_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")
    
    prompt = '''Describe what is in the image in a way that a person could understand. Do not include a description of the black background. Frame your response as a question 
    asking who they are. Here are some examples:
    
    Who is the person with the black hair and hazel eyes?

    Who is the person with the blonde hair, blue eyes, and green hoop earrings?

    NOTE: Respond only with the question and nothing else. Do not add any additional text to your response.
    
    '''

    message = HumanMessage(
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url":f"data:image/png;base64,{image_data}"}}
        ]
    )

    response = "Hello! " + llm.invoke([message]).content

    return response

def ask_for_name(person_img_path: str, use_elevenlabs: bool = False) -> bool:
    description = describe_person(person_img_path)

    if use_elevenlabs:
        audio_stream = voice.text_to_speech.convert_as_stream(
            text=description,
            voice_id="pqHfZKP75CvOlQylNhV4",
            model_id="eleven_multilingual_v2",
            voice_settings=VoiceSettings(
                stability=0.5,
                similarity_boost=0.75,
                use_speaker_boost=True
            )
        )
        stream(audio_stream)
    else:
        engine = pyttsx3.init()
        engine.say(description)
        engine.runAndWait()
        engine.stop()

    return True 

# Function to listen for a person's name
def listen_for_name(use_assemblyai: bool = False, audio_path: str = "audio.wav", save_directory: str = "./outputs/audio") -> str:
    structured_llm = llm.with_structured_output(PersonName)
    
    if use_assemblyai:

        FILE_URL = record_audio(file_name=audio_path, save_directory=save_directory)
        transcript = transcriber.transcribe(FILE_URL)
        if transcript.status == aai.TranscriptStatus.error:
            raise Exception(f"Unable to transcribe audio: {transcript.error}")
        else:
            prompt = "Extract the person's name from the following text: " + transcript.text
            name = structured_llm.invoke(prompt).name
            return name.lower()

    else:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, phrase_time_limit = 5)
        try:
            text =  recognizer.recognize_google(audio)
            prompt = "Extract the person's name from the following text: " + text
            name = structured_llm.invoke(prompt).name
            return name.lower()
        except sr.UnknownValueError as e:
            raise Exception(f"Unable to recognize voice: {e}")

# Function to acknowledge a person's introduction
def acknowledge_person(name:str):

    text = f"Thank you for introducing me to {name}! It's a pleasure to meet you {name}. I will make sure to remember you the next time we meet."
    audio_stream = voice.text_to_speech.convert_as_stream(
        text=text,
        voice_id="pqHfZKP75CvOlQylNhV4",
        model_id="eleven_multilingual_v2",
        voice_settings=VoiceSettings(
            stability=0.5,
            similarity_boost=0.75,
            use_speaker_boost=True
        )
    )

    stream(audio_stream)
    return 

