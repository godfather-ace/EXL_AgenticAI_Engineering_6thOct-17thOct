from autogen_core.models import UserMessage
from autogen_ext.models.ollama import OllamaChatCompletionClient

async def main():
    ollama_model_client = OllamaChatCompletionClient(model="deepseek-r1:1.5b")
    response = await ollama_model_client.create([UserMessage(content="What is the capital of France?", source="user")])
    print(response)
    await ollama_model_client.close()
    
import asyncio
asyncio.run(main())