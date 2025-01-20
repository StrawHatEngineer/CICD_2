import os
from dotenv import load_dotenv
import subprocess

# Load environment variables from .env file
load_dotenv()

# Run the ib-cicd command with environment variables
command = [
    "promote-build-solution", 
    "--vinay"
]

result = subprocess.run(command, env=os.environ)
