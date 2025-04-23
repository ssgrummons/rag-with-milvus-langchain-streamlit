from abc import ABC, abstractmethod
from typing import Optional, AsyncGenerator, Callable
import os
import json
import asyncio
import aiohttp
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APIClientInterface(ABC):
    @abstractmethod
    def get_response(self, user_input: str) -> str:
        pass

    @abstractmethod
    async def get_streaming_response(self, user_input: str, callback: Optional[Callable[[str], None]] = None) -> AsyncGenerator[str, None]:
        pass

class RAGAPIClient(APIClientInterface):
    def __init__(self, base_url: Optional[str] = None, max_retries: int = 3, retry_delay: float = 1.0):
        load_dotenv()
        self.base_url = base_url or os.getenv("API_BASE_URL", "http://localhost:8000")
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def get_response(self, user_input: str) -> str:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._get_response_async(user_input))
        finally:
            loop.close()

    async def get_streaming_response(self, user_input: str, callback: Optional[Callable[[str], None]] = None) -> AsyncGenerator[str, None]:
        async for chunk in self._get_streaming_response_async(user_input, callback):
            yield chunk

    async def _get_response_async(self, user_input: str) -> str:
        url = f"{self.base_url}/chat"
        payload = {"user_prompt": user_input}

        for attempt in range(self.max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload) as response:
                        if response.status == 200:
                            data = await response.json()
                            return data.get("response", "")
                        else:
                            await self._log_and_retry(response, attempt)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.error(f"Request error (attempt {attempt+1}): {str(e)}")
                    await asyncio.sleep(self.retry_delay)
                else:
                    raise Exception(f"Failed to get response after {self.max_retries} attempts: {str(e)}")

    async def _get_streaming_response_async(self, user_input: str, callback: Optional[Callable[[str], None]] = None) -> AsyncGenerator[str, None]:
        url = f"{self.base_url}/chat/stream"
        payload = {"user_prompt": user_input}

        for attempt in range(self.max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload) as response:
                        if response.status == 200:
                            async for chunk in self._handle_response_stream(response, callback):
                                yield chunk
                            return
                        else:
                            await self._log_and_retry(response, attempt)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.error(f"Streaming request error (attempt {attempt+1}): {str(e)}")
                    await asyncio.sleep(self.retry_delay)
                else:
                    raise Exception(f"Failed to get streaming response after {self.max_retries} attempts: {str(e)}")

    async def _handle_response_stream(self, response: aiohttp.ClientResponse, callback: Optional[Callable[[str], None]]) -> AsyncGenerator[str, None]:
        async for line in response.content:
            line = line.decode('utf-8').strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            content_or_error = self._parse_stream_line(data)
            if content_or_error is not None:
                if callback:
                    callback(content_or_error)
                yield content_or_error

    def _parse_stream_line(self, data: str) -> Optional[str]:
        try:
            json_data = json.loads(data)
            if 'content' in json_data:
                return json_data['content']
            elif 'error' in json_data:
                logger.error(f"Streaming error: {json_data['error']}")
                return f"Error: {json_data['error']}"
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON: {data}")
        return None

    async def _log_and_retry(self, response: aiohttp.ClientResponse, attempt: int):
        error_text = await response.text()
        logger.error(f"API error (attempt {attempt+1}/{self.max_retries}): {response.status} - {error_text}")
        if attempt < self.max_retries - 1:
            await asyncio.sleep(self.retry_delay)
        else:
            raise Exception(f"API error: {response.status} - {error_text}")

class APIClientFactory:
    @staticmethod
    def create_client(client_type: str = "rag", **kwargs) -> APIClientInterface:
        if client_type.lower() == "rag":
            return RAGAPIClient(**kwargs)
        raise ValueError(f"Unsupported client type: {client_type}")

def get_response(user_input: str) -> str:
    client = APIClientFactory.create_client()
    return client.get_response(user_input)

async def get_streaming_response(user_input: str, callback: Optional[Callable[[str], None]] = None) -> AsyncGenerator[str, None]:
    client = APIClientFactory.create_client()
    async for chunk in client.get_streaming_response(user_input, callback):
        yield chunk
