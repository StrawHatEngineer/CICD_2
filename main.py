import os
from dotenv import load_dotenv
import subprocess

# Load environment variables from .env file
load_dotenv()

# Run the ib-cicd command with environment variables
command = [
    "promote-build-solution", 
    "--compile_solution",
    "--download_binary",
    # "--get_binary",
    # "--upload_binary",
    # "--publish_advanced_app",
]

result = subprocess.run(command, env=os.environ)
