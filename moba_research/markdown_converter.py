import asyncio
import logging
from datetime import datetime
from typing import Optional

from markitdown import MarkItDown
import httpx

from moba_research.llm_provider import OllamaProvider


class MarkitdownTextReasonerConverter:
    def __init__(self, llm_provider: OllamaProvider, vision_provider, markdown_cleaner_model='deepseek-r1:8b', vision_model='llama3.2-vision:latest'):
        self.llm_provider = llm_provider
        self.markdown_cleaner_model = markdown_cleaner_model
        self.http_client = httpx.AsyncClient(follow_redirects=True)
        self.markitdown = MarkItDown(
            llm_client=vision_provider,
            llm_model=vision_model
        )

    async def convert_to_markdown(self, url: str) -> str:
        """Converts a webpage to Markdown and structures it."""
        try:
            raw_markdown = await asyncio.to_thread(self.markitdown.convert, url)
            if not raw_markdown:
                logging.warn(f"Warning: Empty Markdown from {url}")
                return ""
        except Exception as e:
            logging.error(f"MarkItDown conversion error for {url}: {e}")
            return ""

        # 3. LLM Structuring
        try:
            user_prompt = (f"You are an expert in cleaning up webpage content transformed to markdown. Given the following Markdown content representation of a web page, clean up "
                           f"any irrelevant sections like menus, link farms footers and headers of the original markdown. Out main focus is the main article/section of the page.\n"
                           f"       - Analyze the structure of the markdown to identify what parts can be removed since they are not valuable to answer questions.\n"
                           f"       - Remove the identified irrelevant parts, like link farms, header info or footers.\n"
                           f"       - Keep it as close as possible to the original document.\n"
                           f"       - Don't summarize the markdown. Just remove irrelevant sections.\n"
                           f"       - Don't make up anything, everything on this section should come from the original content.\n"
                           f"Return just a markdown with the cleaned up content.\n"
                           f"---\n"
                           f"Original Article:\n\n{raw_markdown.text_content}")

            messages = [
                {"role": "user", "content": user_prompt}
            ]

            window_size = 16192
            reasoning, response = await self.llm_provider.generate_reasoning(
                messages=messages,
                model=self.markdown_cleaner_model,
                context_window=window_size
            )
            
            if not response:
                logging.warn(f"Warning: Failed to structure content for {url}")
                return raw_markdown.text_content

            return response

        except Exception as e:
            logging.error(f"Structuring error for {url}: {e}")
            return ""

    async def close(self):
        """Close the HTTP client."""
        await self.http_client.aclose()