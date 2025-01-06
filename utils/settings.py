import os
import chromadb
from dotenv import load_dotenv
import assemblyai as aai
from langchain_anthropic import ChatAnthropic
from elevenlabs.client import ElevenLabs

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