from utils.helpers import *

class Hippo:
    def __init__(self, img_path: str,database_path: str, storage_mode = StorageMode.CHROMA):
        self.img_path = img_path
        self.database_path = database_path
        self.storage_mode = storage_mode
        self.img = cv2.cvtColor(cv2.imread(self.img_path), cv2.COLOR_BGR2RGB)
        self.face_states = [FaceState.UNDETECTED, FaceState.UNKNOWN]

    def identify_person(img: np.ndarray, use_elevenlabs: bool = False, use_assemblyai: bool = False, audio_path: str = "audio.wav", save_directory: str = "./outputs/audio") -> tuple[np.ndarray, np.ndarray]:
        print("---------------------------Starting Light Mode---------------------------------")
        database_path = "database/faces"
        print(f"\nUsing database path: {database_path}")
        print("\nSegmenting faces...")
        segmented_faces , face_paths = segment_faces(img)
        if len(segmented_faces) == 0:
            print("No faces detected in the image")
            return False, FaceState.UNDETECTED
        print("\nSegmentation Done!")
        single_face, single_face_path = segmented_faces[0], face_paths[0]
        print(f"\nIdentifying person in {single_face_path}...")

        is_known, possible_name = is_known_face(single_face_path, database_path)
        if is_known:
            print(f"\nPerson already known! :{possible_name}")
            return False, possible_name
        print("\nAsking for person's name")
        name_asked = ask_for_name(single_face_path, use_elevenlabs)
        print("\nAsking Done!")
        if name_asked:
            print(f"\nListening for name...")
            name = listen_for_name(use_assemblyai, audio_path, save_directory)
            print(f"\nName captured as: {name}")
            identity_path = f"{database_path}/{name.lower()}/{name.lower()}.png"
            print(f"\nSaving identity at: {identity_path}")
            os.makedirs(os.path.dirname(identity_path), exist_ok=True)
            cv2.imwrite(identity_path, single_face)
            print("\nIdentity saved!")
            acknowledge_person(name)
            return True, name
        
        return 

