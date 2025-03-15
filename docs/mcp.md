**MCP Python SDK**

**Python implementation of the Model Context Protocol (MCP)**

[](https://camo.githubusercontent.com/e6ba71e25e692956bce8d9b0b4e043d9b7171186941670af455088139928be55/68747470733a2f2f696d672e736869656c64732e696f2f707970692f762f6d63702e737667)

[](https://camo.githubusercontent.com/98147347f1be2b00361083e2aac1a18781acb3109ca688b1cd1940980e9f1201/68747470733a2f2f696d672e736869656c64732e696f2f707970692f6c2f6d63702e737667)

[](https://camo.githubusercontent.com/b33b4fb36a9335985026e9b5b20cf5b1e548b7fff9f215b25abd31c9eaaa04ff/68747470733a2f2f696d672e736869656c64732e696f2f707970692f707976657273696f6e732f6d63702e737667)

[](https://camo.githubusercontent.com/301bdc40b0f2893b417e920988f8aac322e3adab80c8a6c32657286f4aaf3a48/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f646f63732d6d6f64656c636f6e7465787470726f746f636f6c2e696f2d626c75652e737667)

[](https://camo.githubusercontent.com/0e20327998ce56e7a24c9b61227bb10976c5c3b6188551c2bd37e357ad67e7da/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f737065632d737065632e6d6f64656c636f6e7465787470726f746f636f6c2e696f2d626c75652e737667)

[](https://camo.githubusercontent.com/587d3a9857dcc52c6f99b5109e13afc68542ab73eb8160f6a36722bd83a2cb1b/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f64697363757373696f6e732f6d6f64656c636f6e7465787470726f746f636f6c2f707974686f6e2d73646b)

**Table of Contents**

- [Overview](https://github.com/modelcontextprotocol/python-sdk#overview)
- [Installation](https://github.com/modelcontextprotocol/python-sdk#installation)
- [Quickstart](https://github.com/modelcontextprotocol/python-sdk#quickstart)
- [What is MCP?](https://github.com/modelcontextprotocol/python-sdk#what-is-mcp)
- [Core Concepts](https://github.com/modelcontextprotocol/python-sdk#core-concepts)
    - [Server](https://github.com/modelcontextprotocol/python-sdk#server)
    - [Resources](https://github.com/modelcontextprotocol/python-sdk#resources)
    - [Tools](https://github.com/modelcontextprotocol/python-sdk#tools)
    - [Prompts](https://github.com/modelcontextprotocol/python-sdk#prompts)
    - [Images](https://github.com/modelcontextprotocol/python-sdk#images)
    - [Context](https://github.com/modelcontextprotocol/python-sdk#context)
- [Running Your Server](https://github.com/modelcontextprotocol/python-sdk#running-your-server)
    - [Development Mode](https://github.com/modelcontextprotocol/python-sdk#development-mode)
    - [Claude Desktop Integration](https://github.com/modelcontextprotocol/python-sdk#claude-desktop-integration)
    - [Direct Execution](https://github.com/modelcontextprotocol/python-sdk#direct-execution)
- [Examples](https://github.com/modelcontextprotocol/python-sdk#examples)
    - [Echo Server](https://github.com/modelcontextprotocol/python-sdk#echo-server)
    - [SQLite Explorer](https://github.com/modelcontextprotocol/python-sdk#sqlite-explorer)
- [Advanced Usage](https://github.com/modelcontextprotocol/python-sdk#advanced-usage)
    - [Low-Level Server](https://github.com/modelcontextprotocol/python-sdk#low-level-server)
    - [Writing MCP Clients](https://github.com/modelcontextprotocol/python-sdk#writing-mcp-clients)
    - [MCP Primitives](https://github.com/modelcontextprotocol/python-sdk#mcp-primitives)
    - [Server Capabilities](https://github.com/modelcontextprotocol/python-sdk#server-capabilities)
- [Documentation](https://github.com/modelcontextprotocol/python-sdk#documentation)
- [Contributing](https://github.com/modelcontextprotocol/python-sdk#contributing)
- [License](https://github.com/modelcontextprotocol/python-sdk#license)

**Overview**

The Model Context Protocol allows applications to provide context for LLMs in a standardized way, separating the concerns of providing context from the actual LLM interaction. This Python SDK implements the full MCP specification, making it easy to:

- Build MCP clients that can connect to any MCP server
- Create MCP servers that expose resources, prompts and tools
- Use standard transports like stdio and SSE
- Handle all MCP protocol messages and lifecycle events

**Installation**

**Adding MCP to your python project**

We recommend using [uv](https://docs.astral.sh/uv/) to manage your Python projects. In a uv managed python project, add mcp to dependencies by:

```
uv add "mcp[cli]"
```

Alternatively, for projects using pip for dependencies:

```
pip install mcp
```

**Running the standalone MCP development tools**

To run the mcp command with uv:

```
uv run mcp
```

**Quickstart**

Let's create a simple MCP server that exposes a calculator tool and some data:

```
# server.py
from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("Demo")

# Add an addition tool
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

# Add a dynamic greeting resource
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}!"
```

You can install this server in [Claude Desktop](https://claude.ai/download) and interact with it right away by running:

```
mcp install server.py
```

Alternatively, you can test it with the MCP Inspector:

```
mcp dev server.py
```

**What is MCP?**

The [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) lets you build servers that expose data and functionality to LLM applications in a secure, standardized way. Think of it like a web API, but specifically designed for LLM interactions. MCP servers can:

- Expose data through **Resources** (think of these sort of like GET endpoints; they are used to load information into the LLM's context)
- Provide functionality through **Tools** (sort of like POST endpoints; they are used to execute code or otherwise produce a side effect)
- Define interaction patterns through **Prompts** (reusable templates for LLM interactions)
- And more!

**Core Concepts**

**Server**

The FastMCP server is your core interface to the MCP protocol. It handles connection management, protocol compliance, and message routing:

```
# Add lifespan support for startup/shutdown with strong typing
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator

from fake_database import Database  # Replace with your actual DB type

from mcp.server.fastmcp import Context, FastMCP

# Create a named server
mcp = FastMCP("My App")

# Specify dependencies for deployment and development
mcp = FastMCP("My App", dependencies=["pandas", "numpy"])

@dataclassclass AppContext:
    db: Database

@asynccontextmanagerasync def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle with type-safe context"""
    # Initialize on startup
    db = await Database.connect()
    try:
        yield AppContext(db=db)
    finally:
        # Cleanup on shutdown
        await db.disconnect()

# Pass lifespan to server
mcp = FastMCP("My App", lifespan=app_lifespan)

# Access type-safe lifespan context in tools
@mcp.tool()
def query_db(ctx: Context) -> str:
    """Tool that uses initialized resources"""
    db = ctx.request_context.lifespan_context["db"]
    return db.query()
```

**Resources**

Resources are how you expose data to LLMs. They're similar to GET endpoints in a REST API - they provide data but shouldn't perform significant computation or have side effects:

```
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("My App")

@mcp.resource("config://app")
def get_config() -> str:
    """Static configuration data"""
    return "App configuration here"

@mcp.resource("users://{user_id}/profile")
def get_user_profile(user_id: str) -> str:
    """Dynamic user data"""
    return f"Profile data for user {user_id}"
```

**Tools**

Tools let LLMs take actions through your server. Unlike resources, tools are expected to perform computation and have side effects:

```
import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("My App")

@mcp.tool()
def calculate_bmi(weight_kg: float, height_m: float) -> float:
    """Calculate BMI given weight in kg and height in meters"""
    return weight_kg / (height_m**2)

@mcp.tool()
async def fetch_weather(city: str) -> str:
    """Fetch current weather for a city"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://api.weather.com/{city}")
        return response.text
```

**Prompts**

Prompts are reusable templates that help LLMs interact with your server effectively:

```
from mcp.server.fastmcp import FastMCP, types

mcp = FastMCP("My App")

@mcp.prompt()
def review_code(code: str) -> str:
    return f"Please review this code:\n\n{code}"

@mcp.prompt()
def debug_error(error: str) -> list[types.Message]:
    return [
        types.UserMessage("I'm seeing this error:"),
        types.UserMessage(error),
        types.AssistantMessage("I'll help debug that. What have you tried so far?"),
    ]
```

**Images**

FastMCP provides an `Image` class that automatically handles image data:

```
from mcp.server.fastmcp import FastMCP, Image
from PIL import Image as PILImage

mcp = FastMCP("My App")

@mcp.tool()
def create_thumbnail(image_path: str) -> Image:
    """Create a thumbnail from an image"""
    img = PILImage.open(image_path)
    img.thumbnail((100, 100))
    return Image(data=img.tobytes(), format="png")
```

**Context**

The Context object gives your tools and resources access to MCP capabilities:

```
from mcp.server.fastmcp import FastMCP, Context

mcp = FastMCP("My App")

@mcp.tool()
async def long_task(files: list[str], ctx: Context) -> str:
    """Process multiple files with progress tracking"""
    for i, file in enumerate(files):
        ctx.info(f"Processing {file}")
        await ctx.report_progress(i, len(files))
        data, mime_type = await ctx.read_resource(f"file://{file}")
    return "Processing complete"
```

**Running Your Server**

**Development Mode**

The fastest way to test and debug your server is with the MCP Inspector:

```
mcp dev server.py

# Add dependencies
mcp dev server.py --with pandas --with numpy

# Mount local code
mcp dev server.py --with-editable .
```

**Claude Desktop Integration**

Once your server is ready, install it in Claude Desktop:

```
mcp install server.py

# Custom name
mcp install server.py --name "My Analytics Server"# Environment variables
mcp install server.py -v API_KEY=abc123 -v DB_URL=postgres://...
mcp install server.py -f .env
```

**Direct Execution**

For advanced scenarios like custom deployments:

```
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("My App")

if __name__ == "__main__":
    mcp.run()
```

Run it with:

```
python server.py
# or
mcp run server.py
```

**Examples**

**Echo Server**

A simple server demonstrating resources, tools, and prompts:

```
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Echo")

@mcp.resource("echo://{message}")
def echo_resource(message: str) -> str:
    """Echo a message as a resource"""
    return f"Resource echo: {message}"

@mcp.tool()
def echo_tool(message: str) -> str:
    """Echo a message as a tool"""
    return f"Tool echo: {message}"

@mcp.prompt()
def echo_prompt(message: str) -> str:
    """Create an echo prompt"""
    return f"Please process this message: {message}"
```

**SQLite Explorer**

A more complex example showing database integration:

```
import sqlite3

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("SQLite Explorer")

@mcp.resource("schema://main")
def get_schema() -> str:
    """Provide the database schema as a resource"""
    conn = sqlite3.connect("database.db")
    schema = conn.execute("SELECT sql FROM sqlite_master WHERE type='table'").fetchall()
    return "\n".join(sql[0] for sql in schema if sql[0])

@mcp.tool()
def query_data(sql: str) -> str:
    """Execute SQL queries safely"""
    conn = sqlite3.connect("database.db")
    try:
        result = conn.execute(sql).fetchall()
        return "\n".join(str(row) for row in result)
    except Exception as e:
        return f"Error: {str(e)}"
```

**Advanced Usage**

**Low-Level Server**

For more control, you can use the low-level server implementation directly. This gives you full access to the protocol and allows you to customize every aspect of your server, including lifecycle management through the lifespan API:

```
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fake_database import Database  # Replace with your actual DB type

from mcp.server import Server

@asynccontextmanagerasync def server_lifespan(server: Server) -> AsyncIterator[dict]:
    """Manage server startup and shutdown lifecycle."""
    # Initialize resources on startup
    db = await Database.connect()
    try:
        yield {"db": db}
    finally:
        # Clean up on shutdown
        await db.disconnect()

# Pass lifespan to server
server = Server("example-server", lifespan=server_lifespan)

# Access lifespan context in handlers
@server.call_tool()
async def query_db(name: str, arguments: dict) -> list:
    ctx = server.request_context
    db = ctx.lifespan_context["db"]
    return await db.query(arguments["query"])
```

The lifespan API provides:

- A way to initialize resources when the server starts and clean them up when it stops
- Access to initialized resources through the request context in handlers
- Type-safe context passing between lifespan and request handlers

```
import mcp.server.stdio
import mcp.types as types
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.models import InitializationOptions

# Create a server instance
server = Server("example-server")

@server.list_prompts()
async def handle_list_prompts() -> list[types.Prompt]:
    return [
        types.Prompt(
            name="example-prompt",
            description="An example prompt template",
            arguments=[
                types.PromptArgument(
                    name="arg1", description="Example argument", required=True
                )
            ],
        )
    ]

@server.get_prompt()
async def handle_get_prompt(
    name: str, arguments: dict[str, str] | None
) -> types.GetPromptResult:
    if name != "example-prompt":
        raise ValueError(f"Unknown prompt: {name}")

    return types.GetPromptResult(
        description="Example prompt",
        messages=[
            types.PromptMessage(
                role="user",
                content=types.TextContent(type="text", text="Example prompt text"),
            )
        ],
    )

async def run():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="example",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    import asyncio

    asyncio.run(run())
```

**Writing MCP Clients**

The SDK provides a high-level client interface for connecting to MCP servers:

```
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

# Create server parameters for stdio connection
server_params = StdioServerParameters(
    command="python",  # Executable
    args=["example_server.py"],  # Optional command line arguments
    env=None,  # Optional environment variables
)

# Optional: create a sampling callback
async def handle_sampling_message(
    message: types.CreateMessageRequestParams,
) -> types.CreateMessageResult:
    return types.CreateMessageResult(
        role="assistant",
        content=types.TextContent(
            type="text",
            text="Hello, world! from model",
        ),
        model="gpt-3.5-turbo",
        stopReason="endTurn",
    )

async def run():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(
            read, write, sampling_callback=handle_sampling_message
        ) as session:
            # Initialize the connection
            await session.initialize()

            # List available prompts
            prompts = await session.list_prompts()

            # Get a prompt
            prompt = await session.get_prompt(
                "example-prompt", arguments={"arg1": "value"}
            )

            # List available resources
            resources = await session.list_resources()

            # List available tools
            tools = await session.list_tools()

            # Read a resource
            content, mime_type = await session.read_resource("file://some/path")

            # Call a tool
            result = await session.call_tool("tool-name", arguments={"arg1": "value"})

if __name__ == "__main__":
    import asyncio

    asyncio.run(run())
```

**MCP Primitives**

The MCP protocol defines three core primitives that servers can implement:

| Primitive | Control | Description | Example Use |
| --- | --- | --- | --- |
| Prompts | User-controlled | Interactive templates invoked by user choice | Slash commands, menu options |
| Resources | Application-controlled | Contextual data managed by the client application | File contents, API responses |
| Tools | Model-controlled | Functions exposed to the LLM to take actions | API calls, data updates |

**Server Capabilities**

MCP servers declare capabilities during initialization:

| Capability | Feature Flag | Description |
| --- | --- | --- |
| `prompts` | `listChanged` | Prompt template management |
| `resources` | `subscribelistChanged` | Resource exposure and updates |
| `tools` | `listChanged` | Tool discovery and execution |
| `logging` | - | Server logging configuration |
| `completion` | - | Argument completion suggestions |

## Documentation

### **For Client Developers**Get started building your own client that can integrate with all MCP servers.

In this tutorial, you’ll learn how to build a LLM-powered chatbot client that connects to MCP servers. It helps to have gone through the [**Server quickstart**](https://modelcontextprotocol.io/quickstart/server) that guides you through the basic of building your first server.

- **Python**
- **Node**
- **Java**

[**You can find the complete code for this tutorial here.**](https://github.com/modelcontextprotocol/quickstart-resources/tree/main/mcp-client-python)

## **System Requirements**

Before starting, ensure your system meets these requirements:

- Mac or Windows computer
- Latest Python version installed
- Latest version of **`uv`** installed

## **Setting Up Your Environment**

First, create a new Python project with **`uv`**:

Copy

`# Create project directoryuv init mcp-client
cd mcp-client

# Create virtual environmentuv venv

# Activate virtual environment# On Windows:.venv\Scripts\activate
# On Unix or MacOS:source .venv/bin/activate

# Install required packagesuv add mcp anthropic python-dotenv

# Remove boilerplate filesrm hello.py

# Create our main filetouch client.py`

## **Setting Up Your API Key**

You’ll need an Anthropic API key from the [**Anthropic Console**](https://console.anthropic.com/settings/keys).

Create a **`.env`** file to store it:

Copy

`# Create .env filetouch .env`

Add your key to the **`.env`** file:

Copy

`ANTHROPIC_API_KEY=<your key here>`

Add **`.env`** to your **`.gitignore`**:

Copy

`echo ".env" >> .gitignore`

Make sure you keep your **`ANTHROPIC_API_KEY`** secure!

## **Creating the Client**

### **Basic Client Structure**

First, let’s set up our imports and create the basic client class:

Copy

`import asyncio
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .envclass MCPClient:    def __init__(self):        # Initialize session and client objects        self.session: Optional[ClientSession] = None        self.exit_stack = AsyncExitStack()        self.anthropic = Anthropic()    # methods will go here`

### **Server Connection Management**

Next, we’ll implement the method to connect to an MCP server:

Copy

`async def connect_to_server(self, server_script_path: str):    """Connect to an MCP server

    Args:        server_script_path: Path to the server script (.py or .js)    """
    is_python = server_script_path.endswith('.py')    is_js = server_script_path.endswith('.js')    if not (is_python or is_js):        raise ValueError("Server script must be a .py or .js file")    command = "python" if is_python else "node"    server_params = StdioServerParameters(        command=command,        args=[server_script_path],        env=None    )    stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))    self.stdio, self.write = stdio_transport
    self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))    await self.session.initialize()    # List available tools    response = await self.session.list_tools()    tools = response.tools
    print("\nConnected to server with tools:", [tool.name for tool in tools])`

### **Query Processing Logic**

Now let’s add the core functionality for processing queries and handling tool calls:

Copy

`async def process_query(self, query: str) -> str:    """Process a query using Claude and available tools"""    messages = [        {            "role": "user",            "content": query
        }    ]    response = await self.session.list_tools()    available_tools = [{        "name": tool.name,        "description": tool.description,        "input_schema": tool.inputSchema
    } for tool in response.tools]    # Initial Claude API call    response = self.anthropic.messages.create(        model="claude-3-5-sonnet-20241022",        max_tokens=1000,        messages=messages,        tools=available_tools
    )    # Process response and handle tool calls    final_text = []    assistant_message_content = []    for content in response.content:        if content.type == 'text':            final_text.append(content.text)            assistant_message_content.append(content)        elif content.type == 'tool_use':            tool_name = content.name
            tool_args = content.input            # Execute tool call            result = await self.session.call_tool(tool_name, tool_args)            final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")            assistant_message_content.append(content)            messages.append({                "role": "assistant",                "content": assistant_message_content
            })            messages.append({                "role": "user",                "content": [                    {                        "type": "tool_result",                        "tool_use_id": content.id,                        "content": result.content
                    }                ]            })            # Get next response from Claude            response = self.anthropic.messages.create(                model="claude-3-5-sonnet-20241022",                max_tokens=1000,                messages=messages,                tools=available_tools
            )            final_text.append(response.content[0].text)    return "\n".join(final_text)`

### **Interactive Chat Interface**

Now we’ll add the chat loop and cleanup functionality:

Copy

`async def chat_loop(self):    """Run an interactive chat loop"""    print("\nMCP Client Started!")    print("Type your queries or 'quit' to exit.")    while True:        try:            query = input("\nQuery: ").strip()            if query.lower() == 'quit':                break            response = await self.process_query(query)            print("\n" + response)        except Exception as e:            print(f"\nError: {str(e)}")async def cleanup(self):    """Clean up resources"""    await self.exit_stack.aclose()`

### **Main Entry Point**

Finally, we’ll add the main execution logic:

Copy

`async def main():    if len(sys.argv) < 2:        print("Usage: python client.py <path_to_server_script>")        sys.exit(1)    client = MCPClient()    try:        await client.connect_to_server(sys.argv[1])        await client.chat_loop()    finally:        await client.cleanup()if __name__ == "__main__":    import sys
    asyncio.run(main())`

You can find the complete **`client.py`** file [**here.**](https://gist.github.com/zckly/f3f28ea731e096e53b39b47bf0a2d4b1)

## **Key Components Explained**

### **1. Client Initialization**

- The **`MCPClient`** class initializes with session management and API clients
- Uses **`AsyncExitStack`** for proper resource management
- Configures the Anthropic client for Claude interactions

### **2. Server Connection**

- Supports both Python and Node.js servers
- Validates server script type
- Sets up proper communication channels
- Initializes the session and lists available tools

### **3. Query Processing**

- Maintains conversation context
- Handles Claude’s responses and tool calls
- Manages the message flow between Claude and tools
- Combines results into a coherent response

### **4. Interactive Interface**

- Provides a simple command-line interface
- Handles user input and displays responses
- Includes basic error handling
- Allows graceful exit

### **5. Resource Management**

- Proper cleanup of resources
- Error handling for connection issues
- Graceful shutdown procedures

## **Common Customization Points**

1. **Tool Handling**
    - Modify **`process_query()`** to handle specific tool types
    - Add custom error handling for tool calls
    - Implement tool-specific response formatting
2. **Response Processing**
    - Customize how tool results are formatted
    - Add response filtering or transformation
    - Implement custom logging
3. **User Interface**
    - Add a GUI or web interface
    - Implement rich console output
    - Add command history or auto-completion

## **Running the Client**

To run your client with any MCP server:

Copy

`uv run client.py path/to/server.py # python serveruv run client.py path/to/build/index.js # node server`

If you’re continuing the weather tutorial from the server quickstart, your command might look something like this: **`python client.py .../weather/src/weather/server.py`**

The client will:

1. Connect to the specified server
2. List available tools
3. Start an interactive chat session where you can:
    - Enter queries
    - See tool executions
    - Get responses from Claude

Here’s an example of what it should look like if connected to the weather server from the server quickstart:

![](https://mintlify.s3.us-west-1.amazonaws.com/mcp/images/client-claude-cli-python.png)

## **How It Works**

When you submit a query:

1. The client gets the list of available tools from the server
2. Your query is sent to Claude along with tool descriptions
3. Claude decides which tools (if any) to use
4. The client executes any requested tool calls through the server
5. Results are sent back to Claude
6. Claude provides a natural language response
7. The response is displayed to you

## **Best practices**

1. **Error Handling**
    - Always wrap tool calls in try-catch blocks
    - Provide meaningful error messages
    - Gracefully handle connection issues
2. **Resource Management**
    - Use **`AsyncExitStack`** for proper cleanup
    - Close connections when done
    - Handle server disconnections
3. **Security**
    - Store API keys securely in **`.env`**
    - Validate server responses
    - Be cautious with tool permissions

## **Troubleshooting**

### **Server Path Issues**

- Double-check the path to your server script is correct
- Use the absolute path if the relative path isn’t working
- For Windows users, make sure to use forward slashes (/) or escaped backslashes (\) in the path
- Verify the server file has the correct extension (.py for Python or .js for Node.js)

Example of correct path usage:

Copy

`# Relative pathuv run client.py ./server/weather.py

# Absolute pathuv run client.py /Users/username/projects/mcp-server/weather.py

# Windows path (either format works)uv run client.py C:/projects/mcp-server/weather.py
uv run client.py C:\\projects\\mcp-server\\weather.py`

### **Response Timing**

- The first response might take up to 30 seconds to return
- This is normal and happens while:
    - The server initializes
    - Claude processes the query
    - Tools are being executed
- Subsequent responses are typically faster
- Don’t interrupt the process during this initial waiting period

### **Common Error Messages**

If you see:

- **`FileNotFoundError`**: Check your server path
- **`Connection refused`**: Ensure the server is running and the path is correct
- **`Tool execution failed`**: Verify the tool’s required environment variables are set
- **`Timeout error`**: Consider increasing the timeout in your client configuration

### **For Server Developers**Get started building your own server to use in Claude for Desktop and other clients.

In this tutorial, we’ll build a simple MCP weather server and connect it to a host, Claude for Desktop. We’ll start with a basic setup, and then progress to more complex use cases.

### **What we’ll be building**

Many LLMs (including Claude) do not currently have the ability to fetch the forecast and severe weather alerts. Let’s use MCP to solve that!

We’ll build a server that exposes two tools: **`get-alerts`** and **`get-forecast`**. Then we’ll connect the server to an MCP host (in this case, Claude for Desktop):

![](https://mintlify.s3.us-west-1.amazonaws.com/mcp/images/weather-alerts.png)

![](https://mintlify.s3.us-west-1.amazonaws.com/mcp/images/current-weather.png)

Servers can connect to any client. We’ve chosen Claude for Desktop here for simplicity, but we also have guides on [**building your own client**](https://modelcontextprotocol.io/quickstart/client) as well as a [**list of other clients here**](https://modelcontextprotocol.io/clients).

**Why Claude for Desktop and not Claude.ai?**

### **Core MCP Concepts**

MCP servers can provide three main types of capabilities:

1. **Resources**: File-like data that can be read by clients (like API responses or file contents)
2. **Tools**: Functions that can be called by the LLM (with user approval)
3. **Prompts**: Pre-written templates that help users accomplish specific tasks

This tutorial will primarily focus on tools.

- **Python**
- **Node**
- **Java**

Let’s get started with building our weather server! [**You can find the complete code for what we’ll be building here.**](https://github.com/modelcontextprotocol/quickstart-resources/tree/main/weather-server-python)

### **Prerequisite knowledge**

This quickstart assumes you have familiarity with:

- Python
- LLMs like Claude

### **System requirements**

- Python 3.10 or higher installed.
- You must use the Python MCP SDK 1.2.0 or higher.

### **Set up your environment**

First, let’s install **`uv`** and set up our Python project and environment:

**MacOS/LinuxWindows**

Copy

`curl -LsSf https://astral.sh/uv/install.sh | sh`

Make sure to restart your terminal afterwards to ensure that the **`uv`** command gets picked up.

Now, let’s create and set up our project:

**MacOS/LinuxWindows**

Copy

`# Create a new directory for our projectuv init weather
cd weather

# Create virtual environment and activate ituv venv
source .venv/bin/activate

# Install dependenciesuv add "mcp[cli]" httpx

# Create our server filetouch weather.py`

Now let’s dive into building your server.

## **Building your server**

### **Importing packages and setting up the instance**

Add these to the top of your **`weather.py`**:

Copy

`from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP servermcp = FastMCP("weather")# ConstantsNWS_API_BASE = "https://api.weather.gov"USER_AGENT = "weather-app/1.0"`

The FastMCP class uses Python type hints and docstrings to automatically generate tool definitions, making it easy to create and maintain MCP tools.

### **Helper functions**

Next, let’s add our helper functions for querying and formatting the data from the National Weather Service API:

Copy

`async def make_nws_request(url: str) -> dict[str, Any] | None:    """Make a request to the NWS API with proper error handling."""    headers = {        "User-Agent": USER_AGENT,        "Accept": "application/geo+json"    }    async with httpx.AsyncClient() as client:        try:            response = await client.get(url, headers=headers, timeout=30.0)            response.raise_for_status()            return response.json()        except Exception:            return Nonedef format_alert(feature: dict) -> str:    """Format an alert feature into a readable string."""    props = feature["properties"]    return f"""
Event: {props.get('event', 'Unknown')}Area: {props.get('areaDesc', 'Unknown')}Severity: {props.get('severity', 'Unknown')}Description: {props.get('description', 'No description available')}Instructions: {props.get('instruction', 'No specific instructions provided')}"""`

### **Implementing tool execution**

The tool execution handler is responsible for actually executing the logic of each tool. Let’s add it:

Copy

`@mcp.tool()async def get_alerts(state: str) -> str:    """Get weather alerts for a US state.    Args:        state: Two-letter US state code (e.g. CA, NY)    """
    url = f"{NWS_API_BASE}/alerts/active/area/{state}"    data = await make_nws_request(url)    if not data or "features" not in data:        return "Unable to fetch alerts or no alerts found."    if not data["features"]:        return "No active alerts for this state."    alerts = [format_alert(feature) for feature in data["features"]]    return "\n---\n".join(alerts)@mcp.tool()async def get_forecast(latitude: float, longitude: float) -> str:    """Get weather forecast for a location.    Args:        latitude: Latitude of the location
        longitude: Longitude of the location
    """
    # First get the forecast grid endpoint    points_url = f"{NWS_API_BASE}/points/{latitude},{longitude}"    points_data = await make_nws_request(points_url)    if not points_data:        return "Unable to fetch forecast data for this location."    # Get the forecast URL from the points response    forecast_url = points_data["properties"]["forecast"]    forecast_data = await make_nws_request(forecast_url)    if not forecast_data:        return "Unable to fetch detailed forecast."    # Format the periods into a readable forecast    periods = forecast_data["properties"]["periods"]    forecasts = []    for period in periods[:5]:  # Only show next 5 periods        forecast = f"""
{period['name']}:Temperature: {period['temperature']}°{period['temperatureUnit']}Wind: {period['windSpeed']} {period['windDirection']}Forecast: {period['detailedForecast']}"""
        forecasts.append(forecast)    return "\n---\n".join(forecasts)`

### **Running the server**

Finally, let’s initialize and run the server:

Copy

`if __name__ == "__main__":    # Initialize and run the server    mcp.run(transport='stdio')`

Your server is complete! Run **`uv run weather.py`** to confirm that everything’s working.

Let’s now test your server from an existing MCP host, Claude for Desktop.

## **Testing your server with Claude for Desktop**

Claude for Desktop is not yet available on Linux. Linux users can proceed to the [**Building a client**](https://modelcontextprotocol.io/quickstart/client) tutorial to build an MCP client that connects to the server we just built.

First, make sure you have Claude for Desktop installed. [**You can install the latest version here.**](https://claude.ai/download) If you already have Claude for Desktop, **make sure it’s updated to the latest version.**

We’ll need to configure Claude for Desktop for whichever MCP servers you want to use. To do this, open your Claude for Desktop App configuration at **`~/Library/Application Support/Claude/claude_desktop_config.json`** in a text editor. Make sure to create the file if it doesn’t exist.

For example, if you have [**VS Code**](https://code.visualstudio.com/) installed:

- **MacOS/Linux**
- **Windows**

Copy

`code ~/Library/Application\ Support/Claude/claude_desktop_config.json`

You’ll then add your servers in the **`mcpServers`** key. The MCP UI elements will only show up in Claude for Desktop if at least one server is properly configured.

In this case, we’ll add our single weather server like so:

- **MacOS/Linux**
- **Windows**

**PythonCopy**

`{    "mcpServers": {        "weather": {            "command": "uv",            "args": [                "--directory",                "/ABSOLUTE/PATH/TO/PARENT/FOLDER/weather",                "run",                "weather.py"            ]        }    }}`

You may need to put the full path to the **`uv`** executable in the **`command`** field. You can get this by running **`which uv`** on MacOS/Linux or **`where uv`** on Windows.

Make sure you pass in the absolute path to your server.

This tells Claude for Desktop:

1. There’s an MCP server named “weather”
2. To launch it by running **`uv --directory /ABSOLUTE/PATH/TO/PARENT/FOLDER/weather run weather.py`**

Save the file, and restart **Claude for Desktop**.

### **Debugging**A comprehensive guide to debugging Model Context Protocol (MCP) integrations

Effective debugging is essential when developing MCP servers or integrating them with applications. This guide covers the debugging tools and approaches available in the MCP ecosystem.

This guide is for macOS. Guides for other platforms are coming soon.

## **Debugging tools overview**

MCP provides several tools for debugging at different levels:

1. **MCP Inspector**
    - Interactive debugging interface
    - Direct server testing
    - See the [**Inspector guide**](https://modelcontextprotocol.io/docs/tools/inspector) for details
2. **Claude Desktop Developer Tools**
    - Integration testing
    - Log collection
    - Chrome DevTools integration
3. **Server Logging**
    - Custom logging implementations
    - Error tracking
    - Performance monitoring

## **Debugging in Claude Desktop**

### **Checking server status**

The Claude.app interface provides basic server status information:

1. Click the  icon to view:
    
    ![](https://mintlify.s3.us-west-1.amazonaws.com/mcp/images/claude-desktop-mcp-plug-icon.svg)
    
    - Connected servers
    - Available prompts and resources
2. Click the  icon to view:
    
    ![](https://mintlify.s3.us-west-1.amazonaws.com/mcp/images/claude-desktop-mcp-hammer-icon.svg)
    
    - Tools made available to the model

### **Viewing logs**

Review detailed MCP logs from Claude Desktop:

Copy

`# Follow logs in real-timetail -n 20 -F ~/Library/Logs/Claude/mcp*.log`

The logs capture:

- Server connection events
- Configuration issues
- Runtime errors
- Message exchanges

### **Using Chrome DevTools**

Access Chrome’s developer tools inside Claude Desktop to investigate client-side errors:

1. Create a **`developer_settings.json`** file with **`allowDevTools`** set to true:

Copy

`echo '{"allowDevTools": true}' > ~/Library/Application\ Support/Claude/developer_settings.json`

1. Open DevTools: **`Command-Option-Shift-i`**

Note: You’ll see two DevTools windows:

- Main content window
- App title bar window

Use the Console panel to inspect client-side errors.

Use the Network panel to inspect:

- Message payloads
- Connection timing

## **Common issues**

### **Working directory**

When using MCP servers with Claude Desktop:

- The working directory for servers launched via **`claude_desktop_config.json`** may be undefined (like **`/`** on macOS) since Claude Desktop could be started from anywhere
- Always use absolute paths in your configuration and **`.env`** files to ensure reliable operation
- For testing servers directly via command line, the working directory will be where you run the command

For example in **`claude_desktop_config.json`**, use:

Copy

`{  "command": "npx",  "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/username/data"]}`

Instead of relative paths like **`./data`**

### **Environment variables**

MCP servers inherit only a subset of environment variables automatically, like **`USER`**, **`HOME`**, and **`PATH`**.

To override the default variables or provide your own, you can specify an **`env`** key in **`claude_desktop_config.json`**:

Copy

`{  "myserver": {    "command": "mcp-server-myapp",    "env": {      "MYAPP_API_KEY": "some_key",    }  }}`

### **Server initialization**

Common initialization problems:

1. **Path Issues**
    - Incorrect server executable path
    - Missing required files
    - Permission problems
    - Try using an absolute path for **`command`**
2. **Configuration Errors**
    - Invalid JSON syntax
    - Missing required fields
    - Type mismatches
3. **Environment Problems**
    - Missing environment variables
    - Incorrect variable values
    - Permission restrictions

### **Connection problems**

When servers fail to connect:

1. Check Claude Desktop logs
2. Verify server process is running
3. Test standalone with [**Inspector**](https://modelcontextprotocol.io/docs/tools/inspector)
4. Verify protocol compatibility

## **Implementing logging**

### **Server-side logging**

When building a server that uses the local stdio [**transport**](https://modelcontextprotocol.io/docs/concepts/transports), all messages logged to stderr (standard error) will be captured by the host application (e.g., Claude Desktop) automatically.

Local MCP servers should not log messages to stdout (standard out), as this will interfere with protocol operation.

For all [**transports**](https://modelcontextprotocol.io/docs/concepts/transports), you can also provide logging to the client by sending a log message notification:

- **Python**
- **TypeScript**

Copy

`server.request_context.session.send_log_message(  level="info",  data="Server started successfully",)`

Important events to log:

- Initialization steps
- Resource access
- Tool execution
- Error conditions
- Performance metrics

### **Client-side logging**

In client applications:

1. Enable debug logging
2. Monitor network traffic
3. Track message exchanges
4. Record error states

## **Debugging workflow**

### **Development cycle**

1. Initial Development
    - Use [**Inspector**](https://modelcontextprotocol.io/docs/tools/inspector) for basic testing
    - Implement core functionality
    - Add logging points
2. Integration Testing
    - Test in Claude Desktop
    - Monitor logs
    - Check error handling

### **Testing changes**

To test changes efficiently:

- **Configuration changes**: Restart Claude Desktop
- **Server code changes**: Use Command-R to reload
- **Quick iteration**: Use [**Inspector**](https://modelcontextprotocol.io/docs/tools/inspector) during development

## **Best practices**

### **Logging strategy**

1. **Structured Logging**
    - Use consistent formats
    - Include context
    - Add timestamps
    - Track request IDs
2. **Error Handling**
    - Log stack traces
    - Include error context
    - Track error patterns
    - Monitor recovery
3. **Performance Tracking**
    - Log operation timing
    - Monitor resource usage
    - Track message sizes
    - Measure latency

### **Security considerations**

When debugging:

1. **Sensitive Data**
    - Sanitize logs
    - Protect credentials
    - Mask personal information
2. **Access Control**
    - Verify permissions
    - Check authentication
    - Monitor access patterns

## **Getting help**

When encountering issues:

1. **First Steps**
    - Check server logs
    - Test with [**Inspector**](https://modelcontextprotocol.io/docs/tools/inspector)
    - Review configuration
    - Verify environment
2. **Support Channels**
    - GitHub issues
    - GitHub discussions
3. **Providing Information**
    - Log excerpts
    - Configuration files
    - Steps to reproduce
    - Environment details

### **Inspector**In-depth guide to using the MCP Inspector for testing and debugging Model Context Protocol servers

The [**MCP Inspector**](https://github.com/modelcontextprotocol/inspector) is an interactive developer tool for testing and debugging MCP servers. While the [**Debugging Guide**](https://modelcontextprotocol.io/docs/tools/debugging) covers the Inspector as part of the overall debugging toolkit, this document provides a detailed exploration of the Inspector’s features and capabilities.

## **Getting started**

### **Installation and basic usage**

The Inspector runs directly through **`npx`** without requiring installation:

Copy

`npx @modelcontextprotocol/inspector <command>`

Copy

`npx @modelcontextprotocol/inspector <command> <arg1> <arg2>`

### **Inspecting servers from NPM or PyPi**

A common way to start server packages from [**NPM**](https://npmjs.com/) or [**PyPi**](https://pypi.com/).

- **NPM package**
- **PyPi package**

Copy

`npx -y @modelcontextprotocol/inspector npx <package-name> <args># For examplenpx -y @modelcontextprotocol/inspector npx server-postgres postgres://127.0.0.1/testdb`

### **Inspecting locally developed servers**

To inspect servers locally developed or downloaded as a repository, the most common way is:

- **TypeScript**
- **Python**

Copy

`npx @modelcontextprotocol/inspector node path/to/server/index.js args...`

Please carefully read any attached README for the most accurate instructions.

## **Feature overview**

![](https://mintlify.s3.us-west-1.amazonaws.com/mcp/images/mcp-inspector.png)

The MCP Inspector interface

The Inspector provides several features for interacting with your MCP server:

### **Server connection pane**

- Allows selecting the [**transport**](https://modelcontextprotocol.io/docs/concepts/transports) for connecting to the server
- For local servers, supports customizing the command-line arguments and environment

### **Resources tab**

- Lists all available resources
- Shows resource metadata (MIME types, descriptions)
- Allows resource content inspection
- Supports subscription testing

### **Prompts tab**

- Displays available prompt templates
- Shows prompt arguments and descriptions
- Enables prompt testing with custom arguments
- Previews generated messages

### **Tools tab**

- Lists available tools
- Shows tool schemas and descriptions
- Enables tool testing with custom inputs
- Displays tool execution results

### **Notifications pane**

- Presents all logs recorded from the server
- Shows notifications received from the server

## **Best practices**

### **Development workflow**

1. Start Development
    - Launch Inspector with your server
    - Verify basic connectivity
    - Check capability negotiation
2. Iterative testing
    - Make server changes
    - Rebuild the server
    - Reconnect the Inspector
    - Test affected features
    - Monitor messages
3. Test edge cases
    - Invalid inputs
    - Missing prompt arguments
    - Concurrent operations
    - Verify error handling and error responses

### **Prompts**Create reusable prompt templates and workflows

Prompts enable servers to define reusable prompt templates and workflows that clients can easily surface to users and LLMs. They provide a powerful way to standardize and share common LLM interactions.

Prompts are designed to be **user-controlled**, meaning they are exposed from servers to clients with the intention of the user being able to explicitly select them for use.

## **Overview**

Prompts in MCP are predefined templates that can:

- Accept dynamic arguments
- Include context from resources
- Chain multiple interactions
- Guide specific workflows
- Surface as UI elements (like slash commands)

## **Prompt structure**

Each prompt is defined with:

Copy

`{  name: string;              // Unique identifier for the prompt  description?: string;      // Human-readable description  arguments?: [              // Optional list of arguments    {      name: string;          // Argument identifier      description?: string;  // Argument description      required?: boolean;    // Whether argument is required    }  ]}`

## **Discovering prompts**

Clients can discover available prompts through the **`prompts/list`** endpoint:

Copy

`// Request{  method: "prompts/list"}// Response{  prompts: [    {      name: "analyze-code",      description: "Analyze code for potential improvements",      arguments: [        {          name: "language",          description: "Programming language",          required: true        }      ]    }  ]}`

## **Using prompts**

To use a prompt, clients make a **`prompts/get`** request:

Copy

`// Request{  method: "prompts/get",  params: {    name: "analyze-code",    arguments: {      language: "python"    }  }}// Response{  description: "Analyze Python code for potential improvements",  messages: [    {      role: "user",      content: {        type: "text",        text: "Please analyze the following Python code for potential improvements:\n\n```python\ndef calculate_sum(numbers):\n    total = 0\n    for num in numbers:\n        total = total + num\n    return total\n\nresult = calculate_sum([1, 2, 3, 4, 5])\nprint(result)\n```"      }    }  ]}`

## **Dynamic prompts**

Prompts can be dynamic and include:

### **Embedded resource context**

Copy

`{  "name": "analyze-project",  "description": "Analyze project logs and code",  "arguments": [    {      "name": "timeframe",      "description": "Time period to analyze logs",      "required": true    },    {      "name": "fileUri",      "description": "URI of code file to review",      "required": true    }  ]}`

When handling the **`prompts/get`** request:

Copy

`{  "messages": [    {      "role": "user",      "content": {        "type": "text",        "text": "Analyze these system logs and the code file for any issues:"      }    },    {      "role": "user",      "content": {        "type": "resource",        "resource": {          "uri": "logs://recent?timeframe=1h",          "text": "[2024-03-14 15:32:11] ERROR: Connection timeout in network.py:127\n[2024-03-14 15:32:15] WARN: Retrying connection (attempt 2/3)\n[2024-03-14 15:32:20] ERROR: Max retries exceeded",          "mimeType": "text/plain"        }      }    },    {      "role": "user",      "content": {        "type": "resource",        "resource": {          "uri": "file:///path/to/code.py",          "text": "def connect_to_service(timeout=30):\n    retries = 3\n    for attempt in range(retries):\n        try:\n            return establish_connection(timeout)\n        except TimeoutError:\n            if attempt == retries - 1:\n                raise\n            time.sleep(5)\n\ndef establish_connection(timeout):\n    # Connection implementation\n    pass",          "mimeType": "text/x-python"        }      }    }  ]}`

### **Multi-step workflows**

Copy

`const debugWorkflow = {  name: "debug-error",  async getMessages(error: string) {    return [      {        role: "user",        content: {          type: "text",          text: `Here's an error I'm seeing: ${error}`        }      },      {        role: "assistant",        content: {          type: "text",          text: "I'll help analyze this error. What have you tried so far?"        }      },      {        role: "user",        content: {          type: "text",          text: "I've tried restarting the service, but the error persists."        }      }    ];  }};`

## **Example implementation**

Here’s a complete example of implementing prompts in an MCP server:

- **TypeScript**
- **Python**

Copy

`import { Server } from "@modelcontextprotocol/sdk/server";import {  ListPromptsRequestSchema,  GetPromptRequestSchema
} from "@modelcontextprotocol/sdk/types";const PROMPTS = {  "git-commit": {    name: "git-commit",    description: "Generate a Git commit message",    arguments: [      {        name: "changes",        description: "Git diff or description of changes",        required: true      }    ]  },  "explain-code": {    name: "explain-code",    description: "Explain how code works",    arguments: [      {        name: "code",        description: "Code to explain",        required: true      },      {        name: "language",        description: "Programming language",        required: false      }    ]  }};const server = new Server({  name: "example-prompts-server",  version: "1.0.0"}, {  capabilities: {    prompts: {}  }});// List available promptsserver.setRequestHandler(ListPromptsRequestSchema, async () => {  return {    prompts: Object.values(PROMPTS)  };});// Get specific promptserver.setRequestHandler(GetPromptRequestSchema, async (request) => {  const prompt = PROMPTS[request.params.name];  if (!prompt) {    throw new Error(`Prompt not found: ${request.params.name}`);  }  if (request.params.name === "git-commit") {    return {      messages: [        {          role: "user",          content: {            type: "text",            text: `Generate a concise but descriptive commit message for these changes:\n\n${request.params.arguments?.changes}`          }        }      ]    };  }  if (request.params.name === "explain-code") {    const language = request.params.arguments?.language || "Unknown";    return {      messages: [        {          role: "user",          content: {            type: "text",            text: `Explain how this ${language} code works:\n\n${request.params.arguments?.code}`          }        }      ]    };  }  throw new Error("Prompt implementation not found");});`

## **Best practices**

When implementing prompts:

1. Use clear, descriptive prompt names
2. Provide detailed descriptions for prompts and arguments
3. Validate all required arguments
4. Handle missing arguments gracefully
5. Consider versioning for prompt templates
6. Cache dynamic content when appropriate
7. Implement error handling
8. Document expected argument formats
9. Consider prompt composability
10. Test prompts with various inputs

## **UI integration**

Prompts can be surfaced in client UIs as:

- Slash commands
- Quick actions
- Context menu items
- Command palette entries
- Guided workflows
- Interactive forms

## **Updates and changes**

Servers can notify clients about prompt changes:

1. Server capability: **`prompts.listChanged`**
2. Notification: **`notifications/prompts/list_changed`**
3. Client re-fetches prompt list

## **Security considerations**

When implementing prompts:

- Validate all arguments
- Sanitize user input
- Consider rate limiting
- Implement access controls
- Audit prompt usage
- Handle sensitive data appropriately
- Validate generated content
- Implement timeouts
- Consider prompt injection risks
- Document security requirements

### **Sampling**Let your servers request completions from LLMs

Sampling is a powerful MCP feature that allows servers to request LLM completions through the client, enabling sophisticated agentic behaviors while maintaining security and privacy.

This feature of MCP is not yet supported in the Claude Desktop client.

## **How sampling works**

The sampling flow follows these steps:

1. Server sends a **`sampling/createMessage`** request to the client
2. Client reviews the request and can modify it
3. Client samples from an LLM
4. Client reviews the completion
5. Client returns the result to the server

This human-in-the-loop design ensures users maintain control over what the LLM sees and generates.

## **Message format**

Sampling requests use a standardized message format:

Copy

`{  messages: [    {      role: "user" | "assistant",      content: {        type: "text" | "image",        // For text:        text?: string,        // For images:        data?: string,             // base64 encoded        mimeType?: string      }    }  ],  modelPreferences?: {    hints?: [{      name?: string                // Suggested model name/family    }],    costPriority?: number,         // 0-1, importance of minimizing cost    speedPriority?: number,        // 0-1, importance of low latency    intelligencePriority?: number  // 0-1, importance of capabilities  },  systemPrompt?: string,  includeContext?: "none" | "thisServer" | "allServers",  temperature?: number,  maxTokens: number,  stopSequences?: string[],  metadata?: Record<string, unknown>}`

## **Request parameters**

### **Messages**

The **`messages`** array contains the conversation history to send to the LLM. Each message has:

- **`role`**: Either “user” or “assistant”
- **`content`**: The message content, which can be:
    - Text content with a **`text`** field
    - Image content with **`data`** (base64) and **`mimeType`** fields

### **Model preferences**

The **`modelPreferences`** object allows servers to specify their model selection preferences:

- **`hints`**: Array of model name suggestions that clients can use to select an appropriate model:
    - **`name`**: String that can match full or partial model names (e.g. “claude-3”, “sonnet”)
    - Clients may map hints to equivalent models from different providers
    - Multiple hints are evaluated in preference order
- Priority values (0-1 normalized):
    - **`costPriority`**: Importance of minimizing costs
    - **`speedPriority`**: Importance of low latency response
    - **`intelligencePriority`**: Importance of advanced model capabilities

Clients make the final model selection based on these preferences and their available models.

### **System prompt**

An optional **`systemPrompt`** field allows servers to request a specific system prompt. The client may modify or ignore this.

### **Context inclusion**

The **`includeContext`** parameter specifies what MCP context to include:

- **`"none"`**: No additional context
- **`"thisServer"`**: Include context from the requesting server
- **`"allServers"`**: Include context from all connected MCP servers

The client controls what context is actually included.

### **Sampling parameters**

Fine-tune the LLM sampling with:

- **`temperature`**: Controls randomness (0.0 to 1.0)
- **`maxTokens`**: Maximum tokens to generate
- **`stopSequences`**: Array of sequences that stop generation
- **`metadata`**: Additional provider-specific parameters

## **Response format**

The client returns a completion result:

Copy

`{  model: string,  // Name of the model used  stopReason?: "endTurn" | "stopSequence" | "maxTokens" | string,  role: "user" | "assistant",  content: {    type: "text" | "image",    text?: string,    data?: string,    mimeType?: string  }}`

## **Example request**

Here’s an example of requesting sampling from a client:

Copy

`{  "method": "sampling/createMessage",  "params": {    "messages": [      {        "role": "user",        "content": {          "type": "text",          "text": "What files are in the current directory?"        }      }    ],    "systemPrompt": "You are a helpful file system assistant.",    "includeContext": "thisServer",    "maxTokens": 100  }}`

## **Best practices**

When implementing sampling:

1. Always provide clear, well-structured prompts
2. Handle both text and image content appropriately
3. Set reasonable token limits
4. Include relevant context through **`includeContext`**
5. Validate responses before using them
6. Handle errors gracefully
7. Consider rate limiting sampling requests
8. Document expected sampling behavior
9. Test with various model parameters
10. Monitor sampling costs

## **Human in the loop controls**

Sampling is designed with human oversight in mind:

### **For prompts**

- Clients should show users the proposed prompt
- Users should be able to modify or reject prompts
- System prompts can be filtered or modified
- Context inclusion is controlled by the client

### **For completions**

- Clients should show users the completion
- Users should be able to modify or reject completions
- Clients can filter or modify completions
- Users control which model is used

## **Security considerations**

When implementing sampling:

- Validate all message content
- Sanitize sensitive information
- Implement appropriate rate limits
- Monitor sampling usage
- Encrypt data in transit
- Handle user data privacy
- Audit sampling requests
- Control cost exposure
- Implement timeouts
- Handle model errors gracefully

## **Common patterns**

### **Agentic workflows**

Sampling enables agentic patterns like:

- Reading and analyzing resources
- Making decisions based on context
- Generating structured data
- Handling multi-step tasks
- Providing interactive assistance

### **Context management**

Best practices for context:

- Request minimal necessary context
- Structure context clearly
- Handle context size limits
- Update context as needed
- Clean up stale context

### **Error handling**

Robust error handling should:

- Catch sampling failures
- Handle timeout errors
- Manage rate limits
- Validate responses
- Provide fallback behaviors
- Log errors appropriately

## **Limitations**

Be aware of these limitations:

- Sampling depends on client capabilities
- Users control sampling behavior
- Context size has limits
- Rate limits may apply
- Costs should be considered
- Model availability varies
- Response times vary
- Not all content types supported

### **ConceptsRoots**Understanding roots in MCP

Roots are a concept in MCP that define the boundaries where servers can operate. They provide a way for clients to inform servers about relevant resources and their locations.

## **What are Roots?**

A root is a URI that a client suggests a server should focus on. When a client connects to a server, it declares which roots the server should work with. While primarily used for filesystem paths, roots can be any valid URI including HTTP URLs.

For example, roots could be:

Copy

`file:///home/user/projects/myapp
https://api.example.com/v1`

## **Why Use Roots?**

Roots serve several important purposes:

1. **Guidance**: They inform servers about relevant resources and locations
2. **Clarity**: Roots make it clear which resources are part of your workspace
3. **Organization**: Multiple roots let you work with different resources simultaneously

## **How Roots Work**

When a client supports roots, it:

1. Declares the **`roots`** capability during connection
2. Provides a list of suggested roots to the server
3. Notifies the server when roots change (if supported)

While roots are informational and not strictly enforcing, servers should:

1. Respect the provided roots
2. Use root URIs to locate and access resources
3. Prioritize operations within root boundaries

## **Common Use Cases**

Roots are commonly used to define:

- Project directories
- Repository locations
- API endpoints
- Configuration locations
- Resource boundaries

## **Best Practices**

When working with roots:

1. Only suggest necessary resources
2. Use clear, descriptive names for roots
3. Monitor root accessibility
4. Handle root changes gracefully

## **Example**

Here’s how a typical MCP client might expose roots:

Copy

`{  "roots": [    {      "uri": "file:///home/user/projects/frontend",      "name": "Frontend Repository"    },    {      "uri": "https://api.example.com/v1",      "name": "API Endpoint"    }  ]}`

This configuration suggests the server focus on both a local repository and an API endpoint while keeping them logically separated.

### **Tools**Enable LLMs to perform actions through your server

Tools are a powerful primitive in the Model Context Protocol (MCP) that enable servers to expose executable functionality to clients. Through tools, LLMs can interact with external systems, perform computations, and take actions in the real world.

Tools are designed to be **model-controlled**, meaning that tools are exposed from servers to clients with the intention of the AI model being able to automatically invoke them (with a human in the loop to grant approval).

## **Overview**

Tools in MCP allow servers to expose executable functions that can be invoked by clients and used by LLMs to perform actions. Key aspects of tools include:

- **Discovery**: Clients can list available tools through the **`tools/list`** endpoint
- **Invocation**: Tools are called using the **`tools/call`** endpoint, where servers perform the requested operation and return results
- **Flexibility**: Tools can range from simple calculations to complex API interactions

Like [**resources**](https://modelcontextprotocol.io/docs/concepts/resources), tools are identified by unique names and can include descriptions to guide their usage. However, unlike resources, tools represent dynamic operations that can modify state or interact with external systems.

## **Tool definition structure**

Each tool is defined with the following structure:

Copy

`{  name: string;          // Unique identifier for the tool  description?: string;  // Human-readable description  inputSchema: {         // JSON Schema for the tool's parameters    type: "object",    properties: { ... }  // Tool-specific parameters  }}`

## **Implementing tools**

Here’s an example of implementing a basic tool in an MCP server:

- **TypeScript**
- **Python**

Copy

`app = Server("example-server")@app.list_tools()async def list_tools() -> list[types.Tool]:    return [        types.Tool(            name="calculate_sum",            description="Add two numbers together",            inputSchema={                "type": "object",                "properties": {                    "a": {"type": "number"},                    "b": {"type": "number"}                },                "required": ["a", "b"]            }        )    ]@app.call_tool()async def call_tool(    name: str,    arguments: dict) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:    if name == "calculate_sum":        a = arguments["a"]        b = arguments["b"]        result = a + b
        return [types.TextContent(type="text", text=str(result))]    raise ValueError(f"Tool not found: {name}")`

## **Example tool patterns**

Here are some examples of types of tools that a server could provide:

### **System operations**

Tools that interact with the local system:

Copy

`{  name: "execute_command",  description: "Run a shell command",  inputSchema: {    type: "object",    properties: {      command: { type: "string" },      args: { type: "array", items: { type: "string" } }    }  }}`

### **API integrations**

Tools that wrap external APIs:

Copy

`{  name: "github_create_issue",  description: "Create a GitHub issue",  inputSchema: {    type: "object",    properties: {      title: { type: "string" },      body: { type: "string" },      labels: { type: "array", items: { type: "string" } }    }  }}`

### **Data processing**

Tools that transform or analyze data:

Copy

`{  name: "analyze_csv",  description: "Analyze a CSV file",  inputSchema: {    type: "object",    properties: {      filepath: { type: "string" },      operations: {        type: "array",        items: {          enum: ["sum", "average", "count"]        }      }    }  }}`

## **Best practices**

When implementing tools:

1. Provide clear, descriptive names and descriptions
2. Use detailed JSON Schema definitions for parameters
3. Include examples in tool descriptions to demonstrate how the model should use them
4. Implement proper error handling and validation
5. Use progress reporting for long operations
6. Keep tool operations focused and atomic
7. Document expected return value structures
8. Implement proper timeouts
9. Consider rate limiting for resource-intensive operations
10. Log tool usage for debugging and monitoring

## **Security considerations**

When exposing tools:

### **Input validation**

- Validate all parameters against the schema
- Sanitize file paths and system commands
- Validate URLs and external identifiers
- Check parameter sizes and ranges
- Prevent command injection

### **Access control**

- Implement authentication where needed
- Use appropriate authorization checks
- Audit tool usage
- Rate limit requests
- Monitor for abuse

### **Error handling**

- Don’t expose internal errors to clients
- Log security-relevant errors
- Handle timeouts appropriately
- Clean up resources after errors
- Validate return values

## **Tool discovery and updates**

MCP supports dynamic tool discovery:

1. Clients can list available tools at any time
2. Servers can notify clients when tools change using **`notifications/tools/list_changed`**
3. Tools can be added or removed during runtime
4. Tool definitions can be updated (though this should be done carefully)

## **Error handling**

Tool errors should be reported within the result object, not as MCP protocol-level errors. This allows the LLM to see and potentially handle the error. When a tool encounters an error:

1. Set **`isError`** to **`true`** in the result
2. Include error details in the **`content`** array

Here’s an example of proper error handling for tools:

- **TypeScript**
- **Python**

Copy

`try:    # Tool operation    result = perform_operation()    return types.CallToolResult(        content=[            types.TextContent(                type="text",                text=f"Operation successful: {result}"            )        ]    )except Exception as error:    return types.CallToolResult(        isError=True,        content=[            types.TextContent(                type="text",                text=f"Error: {str(error)}"            )        ]    )`

This approach allows the LLM to see that an error occurred and potentially take corrective action or request human intervention.

## **Testing tools**

A comprehensive testing strategy for MCP tools should cover:

- **Functional testing**: Verify tools execute correctly with valid inputs and handle invalid inputs appropriately
- **Integration testing**: Test tool interaction with external systems using both real and mocked dependencies
- **Security testing**: Validate authentication, authorization, input sanitization, and rate limiting
- **Performance testing**: Check behavior under load, timeout handling, and resource cleanup
- **Error handling**: Ensure tools properly report errors through the MCP protocol and clean up resources

### **Transports**Learn about MCP’s communication mechanisms

Transports in the Model Context Protocol (MCP) provide the foundation for communication between clients and servers. A transport handles the underlying mechanics of how messages are sent and received.

## **Message Format**

MCP uses [**JSON-RPC**](https://www.jsonrpc.org/) 2.0 as its wire format. The transport layer is responsible for converting MCP protocol messages into JSON-RPC format for transmission and converting received JSON-RPC messages back into MCP protocol messages.

There are three types of JSON-RPC messages used:

### **Requests**

Copy

`{  jsonrpc: "2.0",  id: number | string,  method: string,  params?: object
}`

### **Responses**

Copy

`{  jsonrpc: "2.0",  id: number | string,  result?: object,  error?: {    code: number,    message: string,    data?: unknown  }}`

### **Notifications**

Copy

`{  jsonrpc: "2.0",  method: string,  params?: object
}`

## **Built-in Transport Types**

MCP includes two standard transport implementations:

### **Standard Input/Output (stdio)**

The stdio transport enables communication through standard input and output streams. This is particularly useful for local integrations and command-line tools.

Use stdio when:

- Building command-line tools
- Implementing local integrations
- Needing simple process communication
- Working with shell scripts
- **TypeScript (Server)**
- **TypeScript (Client)**
- **Python (Server)**
- **Python (Client)**

Copy

`app = Server("example-server")async with stdio_server() as streams:    await app.run(        streams[0],        streams[1],        app.create_initialization_options()    )`

### **Server-Sent Events (SSE)**

SSE transport enables server-to-client streaming with HTTP POST requests for client-to-server communication.

Use SSE when:

- Only server-to-client streaming is needed
- Working with restricted networks
- Implementing simple updates
- **TypeScript (Server)**
- **TypeScript (Client)**
- **Python (Server)**
- **Python (Client)**

Copy

`from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Route

app = Server("example-server")sse = SseServerTransport("/messages")async def handle_sse(scope, receive, send):    async with sse.connect_sse(scope, receive, send) as streams:        await app.run(streams[0], streams[1], app.create_initialization_options())async def handle_messages(scope, receive, send):    await sse.handle_post_message(scope, receive, send)starlette_app = Starlette(    routes=[        Route("/sse", endpoint=handle_sse),        Route("/messages", endpoint=handle_messages, methods=["POST"]),    ])`

## **Custom Transports**

MCP makes it easy to implement custom transports for specific needs. Any transport implementation just needs to conform to the Transport interface:

You can implement custom transports for:

- Custom network protocols
- Specialized communication channels
- Integration with existing systems
- Performance optimization
- **TypeScript**
- **Python**

Note that while MCP Servers are often implemented with asyncio, we recommend implementing low-level interfaces like transports with **`anyio`** for wider compatibility.

Copy

`@contextmanagerasync def create_transport(    read_stream: MemoryObjectReceiveStream[JSONRPCMessage | Exception],    write_stream: MemoryObjectSendStream[JSONRPCMessage]):    """
    Transport interface for MCP.    Args:        read_stream: Stream to read incoming messages from        write_stream: Stream to write outgoing messages to
    """
    async with anyio.create_task_group() as tg:        try:            # Start processing messages            tg.start_soon(lambda: process_messages(read_stream))            # Send messages            async with write_stream:                yield write_stream

        except Exception as exc:            # Handle errors            raise exc
        finally:            # Clean up            tg.cancel_scope.cancel()            await write_stream.aclose()            await read_stream.aclose()`

## **Error Handling**

Transport implementations should handle various error scenarios:

1. Connection errors
2. Message parsing errors
3. Protocol errors
4. Network timeouts
5. Resource cleanup

Example error handling:

- **TypeScript**
- **Python**

Note that while MCP Servers are often implemented with asyncio, we recommend implementing low-level interfaces like transports with **`anyio`** for wider compatibility.

Copy

`@contextmanagerasync def example_transport(scope: Scope, receive: Receive, send: Send):    try:        # Create streams for bidirectional communication        read_stream_writer, read_stream = anyio.create_memory_object_stream(0)        write_stream, write_stream_reader = anyio.create_memory_object_stream(0)        async def message_handler():            try:                async with read_stream_writer:                    # Message handling logic                    pass            except Exception as exc:                logger.error(f"Failed to handle message: {exc}")                raise exc

        async with anyio.create_task_group() as tg:            tg.start_soon(message_handler)            try:                # Yield streams for communication                yield read_stream, write_stream
            except Exception as exc:                logger.error(f"Transport error: {exc}")                raise exc
            finally:                tg.cancel_scope.cancel()                await write_stream.aclose()                await read_stream.aclose()    except Exception as exc:        logger.error(f"Failed to initialize transport: {exc}")        raise exc`

## **Best Practices**

When implementing or using MCP transport:

1. Handle connection lifecycle properly
2. Implement proper error handling
3. Clean up resources on connection close
4. Use appropriate timeouts
5. Validate messages before sending
6. Log transport events for debugging
7. Implement reconnection logic when appropriate
8. Handle backpressure in message queues
9. Monitor connection health
10. Implement proper security measures

## **Security Considerations**

When implementing transport:

### **Authentication and Authorization**

- Implement proper authentication mechanisms
- Validate client credentials
- Use secure token handling
- Implement authorization checks

### **Data Security**

- Use TLS for network transport
- Encrypt sensitive data
- Validate message integrity
- Implement message size limits
- Sanitize input data

### **Network Security**

- Implement rate limiting
- Use appropriate timeouts
- Handle denial of service scenarios
- Monitor for unusual patterns
- Implement proper firewall rules

## **Debugging Transport**

Tips for debugging transport issues:

1. Enable debug logging
2. Monitor message flow
3. Check connection states
4. Validate message formats
5. Test error scenarios
6. Use network analysis tools
7. Implement health checks
8. Monitor resource usage
9. Test edge cases
10. Use proper error tracking