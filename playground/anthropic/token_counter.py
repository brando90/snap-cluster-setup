"""
pip install anthropic-bedrock

ref: https://chatgpt.com/c/cb6f5ecd-1817-43da-b187-94dcae4c5fd7
"""
from anthropic_bedrock import AnthropicBedrock

client = AnthropicBedrock()
prompt = "Hello, world!"
token_count = client.count_tokens(prompt)
print(token_count)

# import anthropic

# client = anthropic.Client()

# token_count = client.count_tokens("Sample text")
# print(token_count)
