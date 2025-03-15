# Create Chat Completion 创建聊天补全。

```
POSThttps://api.deepseek.com/chat/completions
```

Creates a model response for the given chat conversation.

## Request

- application/json

**Bodyrequiredmessages**
object[]**requiredmodel**string**required**
**Possible values:** [`deepseek-chat`, `deepseek-reasoner`]
ID of the model to use. You can use deepseek-chat.**frequency_penalty**number**nullable**
**Possible values:** `>= -2` and `<= 2`
**Default value:** `0`
Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.**max_tokens**integer**nullable**
**Possible values:** `> 1`
Integer between 1 and 8192. The maximum number of tokens that can be generated in the chat completion.
The total length of input tokens and generated tokens is limited by the model's context length.
If `max_tokens` is not specified, the default value 4096 is used.**presence_penalty**number**nullable**
**Possible values:** `>= -2` and `<= 2`
**Default value:** `0`
Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.**response_format**
object**nullable**
An object specifying the format that the model must output. Setting to { "type": "json_object" } enables JSON Output, which guarantees the message the model generates is valid JSON.
**Important:** When using JSON Output, you must also instruct the model to produce JSON yourself via a system or user message. Without this, the model may generate an unending stream of whitespace until the generation reaches the token limit, resulting in a long-running and seemingly "stuck" request. Also note that the message content may be partially cut off if finish_reason="length", which indicates the generation exceeded max_tokens or the conversation exceeded the max context length.**type**string
**Possible values:** [`text`, `json_object`]
**Default value:** `text`
Must be one of `text` or `json_object`.**stop**
object**nullable**
Up to 16 sequences where the API will stop generating further tokens.
oneOf
    ◦ MOD1
    ◦ MOD2
string**stream**boolean**nullable**
If set, partial message deltas will be sent. Tokens will be sent as data-only server-sent events (SSE) as they become available, with the stream terminated by a `data: [DONE]` message.**stream_options**
object**nullable**
Options for streaming response. Only set this when you set `stream: true`.**include_usage**boolean
If set, an additional chunk will be streamed before the `data: [DONE]` message. The `usage` field on this chunk shows the token usage statistics for the entire request, and the `choices` field will always be an empty array. All other chunks will also include a `usage` field, but with a null value.**temperature**number**nullable**
**Possible values:** `<= 2`
**Default value:** `1`
What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
We generally recommend altering this or `top_p` but not both.**top_p**number**nullable**
**Possible values:** `<= 1`
**Default value:** `1`
An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.
We generally recommend altering this or `temperature` but not both.**tools**
object[]**nullable**
A list of tools the model may call. Currently, only functions are supported as a tool. Use this to provide a list of functions the model may generate JSON inputs for. A max of 128 functions are supported.Array [**type**string**required**
**Possible values:** [`function`]
The type of the tool. Currently, only `function` is supported.**function**
object**required**]**tool_choice**
object**nullable**
Controls which (if any) tool is called by the model.
`none` means the model will not call any tool and instead generates a message.
`auto` means the model can pick between generating a message or calling one or more tools.
`required` means the model must call one or more tools.
Specifying a particular tool via `{"type": "function", "function": {"name": "my_function"}}` forces the model to call that tool.
`none` is the default when no tools are present. `auto` is the default if tools are present.
oneOf
    ◦ ChatCompletionToolChoice
    ◦ ChatCompletionNamedToolChoice
string
**Possible values:** [`none`, `auto`, `required`]**logprobs**boolean**nullable**
Whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each output token returned in the `content` of `message`.**top_logprobs**integer**nullable**
**Possible values:** `<= 20`
An integer between 0 and 20 specifying the number of most likely tokens to return at each token position, each with an associated log probability. `logprobs` must be set to `true` if this parameter is used.

- ◦ MOD1
- ◦ MOD2
- Array [
- ]
- ◦ ChatCompletionToolChoice
- ◦ ChatCompletionNamedToolChoice

## Responses

- 200 (No streaming)
- 200 (Streaming)

OK, returns a `chat completion object`

- application/json
- Schema
- Example (from schema)
- Example

**Schemaid**string**required**
A unique identifier for the chat completion.**choices**
object[]**requiredcreated**integer**required**
The Unix timestamp (in seconds) of when the chat completion was created.**model**string**required**
The model used for the chat completion.**system_fingerprint**string**required**
This fingerprint represents the backend configuration that the model runs with.**object**string**required**
**Possible values:** [`chat.completion`]
The object type, which is always `chat.completion`.**usage**
object
Usage statistics for the completion request.**completion_tokens**integer**required**
Number of tokens in the generated completion.**prompt_tokens**integer**required**
Number of tokens in the prompt. It equals prompt_cache_hit_tokens + prompt_cache_miss_tokens.**prompt_cache_hit_tokens**integer**required**
Number of tokens in the prompt that hits the context cache.**prompt_cache_miss_tokens**integer**required**
Number of tokens in the prompt that misses the context cache.**total_tokens**integer**required**
Total number of tokens used in the request (prompt + completion).**completion_tokens_details**
object

- curl
- python
- go
- nodejs
- ruby
- csharp
- php
- java
- powershell
- OpenAI SDK

```python
from openai import OpenAI

# for backward compatibility, you can still use `https://api.deepseek.com/v1` as `base_url`.
client = OpenAI(api_key="<your API key>", base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
  ],
    max_tokens=1024,
    temperature=0.7,
    stream=False
)

print(response.choices[0].message.content)

```

- REQUESTS
- HTTP.CLIENT

```python
import requests
import json

url = "https://api.deepseek.com/chat/completions"

payload = json.dumps({
  "messages": [
    {
      "content": "You are a helpful assistant",
      "role": "system"
    },
    {
      "content": "Hi",
      "role": "user"
    }
  ],
  "model": "deepseek-chat",
  "frequency_penalty": 0,
  "max_tokens": 2048,
  "presence_penalty": 0,
  "response_format": {
    "type": "text"
  },
  "stop": None,
  "stream": False,
  "stream_options": None,
  "temperature": 1,
  "top_p": 1,
  "tools": None,
  "tool_choice": "none",
  "logprobs": False,
  "top_logprobs": None
})
headers = {
  'Content-Type': 'application/json',
  'Accept': 'application/json',
  'Authorization': 'Bearer <TOKEN>'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)

```

**RequestCollapse all**Base URLhttps://api.deepseek.comAuth**Bearer Token**Body **required**

`{
  "messages": [
    {
      "content": "You are a helpful assistant",
      "role": "system"
    },
    {
      "content": "Hi",
      "role": "user"
    }
  ],
  "model": "deepseek-chat",
  "frequency_penalty": 0,
  "max_tokens": 2048,
  "presence_penalty": 0,
  "response_format": {
    "type": "text"
  },
  "stop": null,
  "stream": false,
  "stream_options": null,
  "temperature": 1,
  "top_p": 1,
  "tools": null,
  "tool_choice": "none",
  "logprobs": false,
  "top_logprobs": null
}`
**Send API Request**

**ResponseClear**

Click the `Send API Request` button above and see the response here!

[PreviousIntroduction](https://api-docs.deepseek.com/api/deepseek-api)

# Chat Prefix Completion (Beta)

The chat prefix completion follows the [Chat Completion API](https://api-docs.deepseek.com/api/create-chat-completion), where users provide an assistant's prefix message for the model to complete the rest of the message.

## Notice

1. When using chat prefix completion, users must ensure that the `role` of the last message in the `messages` list is `assistant` and set the `prefix` parameter of the last message to `True`.
2. The user needs to set `base_url="https://api.deepseek.com/beta"` to enable the Beta feature.

## Sample Code

Below is a complete Python code example for chat prefix completion. In this example, we set the prefix message of the `assistant` to `"```python\n"` to force the model to output Python code, and set the `stop` parameter to `['```']` to prevent additional explanations from the model.

```python
from openai import OpenAI

client = OpenAI(
    api_key="<your api key>",
    base_url="https://api.deepseek.com/beta",
)

messages = [
    {"role": "user", "content": "Please write quick sort code"},
    {"role": "assistant", "content": "```python\n", "prefix": True}
]
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages,
    stop=["```"],
)
print(response.choices[0].message.content)

```