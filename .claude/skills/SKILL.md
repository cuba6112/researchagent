---
name: gemini-genai
description: Google python-genai SDK for Gemini 3 Flash, Gemini 3 Pro, and other Gemini models. Use when building applications with Google's Gemini API, Google AI, google-genai, implementing chat, structured outputs, thinking/reasoning, streaming, multimodal inputs (images, video, audio, PDF), function calling, or file uploads. Triggers on "gemini", "google ai", "genai", "google llm". Covers both Gemini Developer API and Vertex AI backends.
---

# Google Gemini python-genai SDK Guide

Build applications with Google's official python-genai SDK for Gemini models.

## Installation

```bash
pip install google-genai
```

## Quick Start

```python
from google import genai
from google.genai import types

# Initialize client (uses GEMINI_API_KEY env var)
client = genai.Client()

# Or explicit API key
client = genai.Client(api_key='YOUR_API_KEY')

# Generate content
response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="Explain quantum computing"
)
print(response.text)
```

## Model IDs

### Gemini 3 Series (Latest)
| Model | ID | Best For |
|-------|-----|----------|
| Gemini 3 Flash | `gemini-3-flash-preview` | Fast responses with thinking |
| Gemini 3 Pro | `gemini-3-pro-preview` | Complex reasoning |
| Gemini 3 Pro Image | `gemini-3-pro-image-preview` | Image generation |

### Gemini 2.5 Series (Stable)
| Model | ID | Best For |
|-------|-----|----------|
| Gemini 2.5 Flash | `gemini-2.5-flash` | Price-performance balance |
| Gemini 2.5 Pro | `gemini-2.5-pro` | Advanced reasoning |

### Token Limits
- **Input**: 1M tokens | **Output**: 64k tokens

## Client Configuration

### Gemini Developer API
```python
from google import genai

# Via environment variable (recommended)
# export GEMINI_API_KEY='your-key'
client = genai.Client()

# Or explicit
client = genai.Client(api_key='YOUR_API_KEY')
```

### Vertex AI
```python
client = genai.Client(
    vertexai=True,
    project='your-project-id',
    location='us-central1'
)
```

### Context Manager (Recommended)
```python
with genai.Client() as client:
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents="Hello!"
    )
```

## Thinking Levels (Gemini 3)

| Level | Use Case |
|-------|----------|
| `minimal` | No thinking, lowest latency (Flash only) |
| `low` | Simple tasks, chat |
| `medium` | Balanced (Flash only) |
| `high` | Maximum reasoning (default) |

```python
from google.genai import types

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="Solve this step by step: ...",
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level="high")
    ),
)
```

**Important:** Keep `temperature=1.0` for Gemini 3. Lower values may cause looping.

## Streaming

```python
for chunk in client.models.generate_content_stream(
    model="gemini-3-flash-preview",
    contents="Tell me a story"
):
    print(chunk.text, end="", flush=True)
```

## Async Operations

```python
import asyncio
from google import genai

async def main():
    async with genai.Client().aio as client:
        response = await client.models.generate_content(
            model="gemini-3-flash-preview",
            contents="Explain AI"
        )
        print(response.text)

asyncio.run(main())
```

### Async Streaming
```python
async for chunk in await client.aio.models.generate_content_stream(
    model="gemini-3-flash-preview",
    contents="Tell me a story"
):
    print(chunk.text, end="")
```

## Chat Sessions

Use `client.chats.create()` for multi-turn conversations (handles history automatically):

```python
chat = client.chats.create(model="gemini-3-flash-preview")

response = chat.send_message("Hi, I'm learning Python")
print(response.text)

response = chat.send_message("What should I learn first?")
print(response.text)
```

### Streaming Chat
```python
chat = client.chats.create(model="gemini-3-flash-preview")

for chunk in chat.send_message_stream("Tell me a story"):
    print(chunk.text, end="")
```

### Async Chat
```python
chat = client.aio.chats.create(model="gemini-3-flash-preview")
response = await chat.send_message("Hello!")
```

## Multimodal Inputs

### Image from File
```python
from google.genai import types

with open("image.jpg", "rb") as f:
    image_bytes = f.read()

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=[
        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
        "Describe this image"
    ]
)
```

### Image from GCS
```python
response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=[
        types.Part.from_uri(
            file_uri="gs://bucket/image.jpg",
            mime_type="image/jpeg"
        ),
        "What's in this image?"
    ]
)
```

### Media Resolution (Per-Part)
```python
from google.genai import types

# Resolution is set on the Part, not in config
# Requires v1alpha API
client = genai.Client(http_options=types.HttpOptions(api_version='v1alpha'))

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=[
        types.Content(
            parts=[
                types.Part.from_text("What's in this image?"),
                types.Part(
                    inline_data=types.Blob(mime_type="image/jpeg", data=image_bytes),
                    media_resolution={"level": "media_resolution_high"}
                )
            ]
        )
    ]
)
```

Resolution tokens: `low`=280, `medium`=560, `high`=1120, `ultra_high` (images only)

## File Upload (Gemini Developer API)

```python
file = client.files.upload(file='document.pdf')

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=["Summarize this document:", file]
)
```

## Structured Outputs

### JSON Response
```python
response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="List 3 programming languages",
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
    ),
)
print(response.text)  # Raw JSON string
```

### With JSON Schema
```python
schema = {
    "type": "object",
    "properties": {
        "languages": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "use_case": {"type": "string"}
                },
                "required": ["name", "use_case"]
            }
        }
    },
    "required": ["languages"]
}

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="List 3 programming languages with use cases",
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_json_schema=schema,
    ),
)
print(response.parsed)  # Auto-parsed dict
```

### With Pydantic (Recommended)
```python
from pydantic import BaseModel
from typing import List

class Language(BaseModel):
    name: str
    use_case: str

class LanguageList(BaseModel):
    languages: List[Language]

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="List 3 programming languages with use cases",
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=LanguageList,
    ),
)
result = LanguageList.model_validate_json(response.text)
```

### Enum Response
```python
from enum import Enum

class Instrument(Enum):
    PERCUSSION = "Percussion"
    STRING = "String"
    KEYBOARD = "Keyboard"

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="What instrument plays multiple notes at once?",
    config={
        "response_mime_type": "text/x.enum",
        "response_schema": Instrument,
    },
)
print(response.text)  # "Keyboard"
```

## Function Calling

### Automatic (Python Functions)
```python
def get_weather(location: str) -> str:
    """Get current weather for a location.
    
    Args:
        location: City and state, e.g. "San Francisco, CA"
    """
    return f"Sunny in {location}"

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="What's the weather in Boston?",
    config=types.GenerateContentConfig(tools=[get_weather]),
)
print(response.text)  # Auto-called and responded
```

### Disable Auto-Calling
```python
response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="What's the weather in Boston?",
    config=types.GenerateContentConfig(
        tools=[get_weather],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            disable=True
        ),
    ),
)
# Access function calls
for fc in response.function_calls:
    print(f"{fc.name}: {fc.args}")
```

### Manual Declaration
```python
weather_tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="get_weather",
            description="Get current weather for a location",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name"
                    }
                },
                "required": ["location"]
            }
        )
    ]
)

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="What's the weather in Tokyo?",
    config=types.GenerateContentConfig(tools=[weather_tool]),
)

# Check for function calls
if response.function_calls:
    fc = response.function_calls[0]
    print(f"Call: {fc.name}({fc.args})")
```

### Send Function Response
```python
# After getting function call, send result back
function_response_part = types.Part.from_function_response(
    name="get_weather",
    response={"result": "Sunny, 72°F"}
)

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=[
        user_message,
        model_function_call_content,
        types.Content(role="tool", parts=[function_response_part])
    ],
    config=types.GenerateContentConfig(tools=[weather_tool]),
)
```

## Built-in Tools (Gemini 3)

### Google Search
```python
response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="Latest AI news",
    config=types.GenerateContentConfig(
        tools=[{"google_search": {}}],
    ),
)
```

### URL Context
```python
response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="Summarize this page: https://example.com/article",
    config=types.GenerateContentConfig(
        tools=[{"url_context": {}}],
    ),
)
```

### Code Execution
```python
response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="Calculate first 10 Fibonacci numbers",
    config=types.GenerateContentConfig(
        tools=[{"code_execution": {}}],
    ),
)
```

### Combine Tools with Structured Output
```python
from pydantic import BaseModel
from typing import List

class SearchResult(BaseModel):
    headline: str
    summary: str

class NewsResults(BaseModel):
    results: List[SearchResult]

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="Find latest AI headlines",
    config={
        "tools": [{"google_search": {}}],
        "response_mime_type": "application/json",
        "response_json_schema": NewsResults.model_json_schema(),
    },
)
```

## Image Generation (Gemini 3 Pro Image)

```python
response = client.models.generate_content(
    model="gemini-3-pro-image-preview",
    contents="A cyberpunk city at sunset",
    config=types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        image_config=types.ImageConfig(
            aspect_ratio="16:9",
            image_size="4K"  # or "2K"
        ),
    ),
)

for part in response.parts:
    if part.inline_data:
        image = part.as_image()
        image.save("generated.png")
```

### Grounded Image Generation
```python
response = client.models.generate_content(
    model="gemini-3-pro-image-preview",
    contents="Infographic of current weather in Tokyo",
    config=types.GenerateContentConfig(
        tools=[{"google_search": {}}],
        response_modalities=["IMAGE"],
    ),
)
```

## Error Handling

```python
from google.genai import errors

try:
    response = client.models.generate_content(
        model="invalid-model",
        contents="Hello"
    )
except errors.APIError as e:
    print(f"Error {e.code}: {e.message}")
```

## Best Practices for Gemini 3

1. **Keep temperature=1.0** - Lower values cause looping
2. **Be concise** - Direct instructions work best
3. **Place instructions at end** - After data context
4. **Use appropriate thinking_level** - `low` for simple, `high` for complex
5. **SDK handles thought signatures** - No manual management needed
6. **Use `response.function_calls`** - Cleaner than drilling into candidates
7. **Use `response.parsed`** - Auto-parsed JSON responses

## Pricing (Gemini 3 Flash)

- **Input**: $0.50 / 1M tokens
- **Output**: $3.00 / 1M tokens
- **Free tier**: Available
- **Context caching**: Up to 90% cost reduction

## Resources

- [python-genai GitHub](https://github.com/googleapis/python-genai)
- [SDK Documentation](https://googleapis.github.io/python-genai/)
- [Gemini 3 Guide](https://ai.google.dev/gemini-api/docs/gemini-3)
