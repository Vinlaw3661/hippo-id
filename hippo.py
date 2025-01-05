
import os
import cv2
import numpy as np 
from utils.helpers import (
    segment_faces,
    is_known_face,
    ask_for_name,
    listen_for_name,
    acknowledge_person,
    StorageMode,
    FaceState
)



class Hippo:
    def __init__(self,database_path: str = "chroma", storage_mode: StorageMode = StorageMode.CHROMA, use_elevenlabs: bool = False, use_assemblyai: bool = False, audio_path: str = "audio.wav", audio_save_directory: str = "./outputs/audio"):
        self.database_path = database_path
        self.storage_mode = storage_mode
        self.use_elevenlabs = use_elevenlabs
        self.use_assemblyai = use_assemblyai
        self.audio_path = audio_path
        self.audio_save_directory = audio_save_directory
        self.face_states = [FaceState.UNDETECTED, FaceState.UNKNOWN]

    def identify(self, img) -> tuple[np.ndarray, np.ndarray]:
        print("---------------------------Starting Light Mode---------------------------------")
        database_path = "database/faces"
        print(f"\nUsing database path: {database_path}")
        print("\nSegmenting faces...")
        segmented_face , face_path, face_state = segment_faces(self.img)
        if face_state == FaceState.UNDETECTED:
            print("No faces detected in the image")
            return False, FaceState.UNDETECTED
        print("\nSegmentation Done!")
        single_face, single_face_path = segmented_face, face_path
        print(f"\nIdentifying person in {single_face_path}...")

        is_known, possible_name = is_known_face(single_face_path, database_path)
        if is_known:
            print(f"\nPerson already known! :{possible_name}")
            return False, possible_name
        print("\nAsking for person's name")
        name_asked = ask_for_name(single_face_path, self.use_elevenlabs)
        print("\nAsking Done!")
        if name_asked:
            print(f"\nListening for name...")
            name = listen_for_name(self.use_assemblyai, self.audio_path, self.audio_save_directory)
            print(f"\nName captured as: {name}")
            identity_path = f"{database_path}/{name.lower()}/{name.lower()}.png"
            print(f"\nSaving identity at: {identity_path}")
            os.makedirs(os.path.dirname(identity_path), exist_ok=True)
            cv2.imwrite(identity_path, single_face)
            print("\nIdentity saved!")
            acknowledge_person(name)
            return True, name
        
        else:
            raise Exception("Unable to ask for person's name")

