
import os
import cv2
import numpy as np 
from utils.helpers import (
    verbose_print,
    segment_faces,
    is_known_face,
    is_known_face_deepface,
    ask_for_name,
    listen_for_name,
    acknowledge_person,
    store_face_embedding,
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

    def identify(self, image, verbose: bool = True) -> tuple[np.ndarray, np.ndarray]:

        verbose_print("---------------------------Starting Identification---------------------------------")
        verbose_print(f"\nUsing database path: {self.database_path}")
        verbose_print("\nSegmenting faces...")

        segmented_face , face_path, face_state = segment_faces(image)

        if face_state == FaceState.UNDETECTED:

            verbose_print("No faces detected in the image")
            return False, FaceState.UNDETECTED
        
        verbose_print("\nSegmentation Done!")

        single_face, single_face_path = segmented_face, face_path
        verbose_print(f"\nIdentifying person in {single_face_path}...")

        if self.storage_mode == StorageMode.CHROMA:
            is_known, possible_name = is_known_face(single_face_path)

        elif self.storage_mode == StorageMode.DEEPFACE:
            is_known, possible_name = is_known_face_deepface(single_face_path, self.database_path)

        if is_known:
            verbose_print(f"\nPerson already known! :{possible_name}")
            return False, possible_name
        
        verbose_print("\nAsking for person's name")
        name_asked = ask_for_name(single_face_path, self.use_elevenlabs)
        verbose_print("\nAsking Done!")

        if name_asked:
            verbose_print(f"\nListening for name...")
            name = listen_for_name(self.use_assemblyai, self.audio_path, self.audio_save_directory)
            verbose_print(f"\nName captured as: {name}")

            if self.storage_mode == StorageMode.CHROMA:
                verbose_print(f"\nSaving identity to chromadatabase: {self.database_path}")
                store_face_embedding(single_face_path, name)
            
            elif self.storage_mode == StorageMode.DEEPFACE:
                identity_path = f"{self.database_path}/{name.lower()}/{name.lower()}.png"
                verbose_print(f"\nSaving identity at: {identity_path}")
                os.makedirs(os.path.dirname(identity_path), exist_ok=True)
                cv2.imwrite(identity_path, single_face)

            verbose_print("\nIdentity saved!")
            acknowledge_person(name)
            return True, name
        
        else:
            raise Exception("Unable to ask for person's name")

