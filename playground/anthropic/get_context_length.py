import os
from anthropic import Anthropic

# Set your API key as an environment variable
os.environ['ANTHROPIC_API_KEY'] = 'your-api-key-here'

# Initialize the Anthropic client
client = Anthropic()

# Define the conversation
conversation = [
    {"role": "user", "content": "Explain quantum computing."},
    {"role": "assistant", "content": "Quantum computing is..."},
    {"role": "user", "content": "Provide sources for this explanation."}
]

# Send the request to the API
response = client.messages.create(
    model="claude-2.1",
    max_tokens=512,
    messages=conversation
)

# Extract and print context links from the response
context_links = response.get('context_links', [])
for link in context_links:
    print(link)
