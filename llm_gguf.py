import click
import httpx
import json
import re
from llama_cpp import Llama
from llama_cpp import llama_chat_format
import llm
import pathlib


def _ensure_models_dir():
    directory = llm.user_dir() / "gguf" / "models"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _ensure_models_file():
    directory = llm.user_dir() / "gguf"
    directory.mkdir(parents=True, exist_ok=True)
    filepath = directory / "models.json"
    if not filepath.exists():
        filepath.write_text("{}")
    return filepath


def _ensure_embed_models_file():
    directory = llm.user_dir() / "gguf"
    directory.mkdir(parents=True, exist_ok=True)
    filepath = directory / "embed-models.json"
    if not filepath.exists():
        filepath.write_text("{}")
    return filepath


@llm.hookimpl
def register_models(register):
    models_file = _ensure_models_file()
    models = json.loads(models_file.read_text())
    for model_id, info in models.items():
        model_path = info["path"]
        aliases = info.get("aliases", [])
        clip_model_path = info.get("clip_model_path")
        chat_handler_class = info.get("chat_handler_class")
        model_id = f"gguf/{model_id}"
        model = GgufChatModel(
            model_id,
            model_path,
            clip_model_path=clip_model_path,
            chat_handler_class=chat_handler_class,
            n_ctx=info.get("n_ctx", 0),
        )
        register(model, aliases=aliases)


@llm.hookimpl
def register_embedding_models(register):
    models_file = _ensure_embed_models_file()
    models = json.loads(models_file.read_text())
    for model_id, info in models.items():
        model_path = info["path"]
        aliases = info.get("aliases", [])
        model_id = f"gguf/{model_id}"
        model = GgufEmbeddingModel(model_id, model_path)
        register(model, aliases=aliases)


@llm.hookimpl
def register_commands(cli):
    @cli.group()
    def gguf():
        "Commands for working with GGUF models"

    @gguf.command()
    def models_file():
        "Display the path to the gguf/models.json file"
        directory = llm.user_dir() / "gguf"
        directory.mkdir(parents=True, exist_ok=True)
        models_file = directory / "models.json"
        click.echo(models_file)

    @gguf.command()
    def embed_models_file():
        "Display the path to the gguf/embed-models.json file"
        directory = llm.user_dir() / "gguf"
        directory.mkdir(parents=True, exist_ok=True)
        models_file = directory / "embed-models.json"
        click.echo(models_file)

    @gguf.command()
    def models_dir():
        "Display the path to the directory holding downloaded GGUF models"
        click.echo(_ensure_models_dir())

    @gguf.command()
    @click.argument("url")
    @click.option(
        "aliases",
        "-a",
        "--alias",
        multiple=True,
        help="Alias(es) to register the model under",
    )
    def download_model(url, aliases):
        "Download and register a GGUF model from a URL"
        download_gguf_model(url, _ensure_models_file, aliases)

    @gguf.command()
    @click.argument("url")
    @click.option(
        "aliases",
        "-a",
        "--alias",
        multiple=True,
        help="Alias(es) to register the model under",
    )
    def download_embed_model(url, aliases):
        "Download and register a GGUF embedding model from a URL"
        download_gguf_model(url, _ensure_embed_models_file, aliases)

    @gguf.command()
    @click.argument("model_id")
    @click.argument(
        "filepath", type=click.Path(exists=True, dir_okay=False, resolve_path=True)
    )
    @click.option(
        "clip_model_path",
        "--clip-model-path",
        type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    )
    @click.option("chat_handler_class", "--chat-handler-class", type=str)
    @click.option("n_ctx", "--n-ctx", type=int, default=0)
    @click.option(
        "aliases",
        "-a",
        "--alias",
        multiple=True,
        help="Alias(es) to register the model under",
    )
    def register_model(
        model_id, filepath, clip_model_path, chat_handler_class, n_ctx, aliases
    ):
        "Register a GGUF model that you have already downloaded with LLM"
        models_file = _ensure_models_file()
        models = json.loads(models_file.read_text())
        path = pathlib.Path(filepath)
        info = {
            "path": str(path.resolve()),
            "aliases": aliases,
        }
        if clip_model_path:
            info["clip_model_path"] = clip_model_path
        if chat_handler_class:
            if not hasattr(llama_chat_format, chat_handler_class):
                raise click.ClickException(
                    f"Invalid chat handler class: {chat_handler_class}"
                )
            info["chat_handler_class"] = chat_handler_class
        if n_ctx:
            info["n_ctx"] = n_ctx
        models[model_id] = info
        models_file.write_text(json.dumps(models, indent=2))

    @gguf.command()
    @click.argument("model_id")
    @click.argument(
        "filepath", type=click.Path(exists=True, dir_okay=False, resolve_path=True)
    )
    @click.option(
        "aliases",
        "-a",
        "--alias",
        multiple=True,
        help="Alias(es) to register the model under",
    )
    def register_embed_model(model_id, filepath, aliases):
        "Register a GGUF embedding model that you have already downloaded"
        models_file = _ensure_embed_models_file()
        models = json.loads(models_file.read_text())
        path = pathlib.Path(filepath)
        info = {
            "path": str(path.resolve()),
            "aliases": aliases,
        }
        models[model_id] = info
        models_file.write_text(json.dumps(models, indent=2))

    @gguf.command()
    def models():
        "List registered GGUF models"
        models_file = _ensure_models_file()
        models = json.loads(models_file.read_text())
        for model, info in models.items():
            try:
                info["size"] = human_size(pathlib.Path(info["path"]).stat().st_size)
            except FileNotFoundError:
                info["size"] = None
        click.echo(json.dumps(models, indent=2))

    @gguf.command()
    def embed_models():
        "List registered GGUF embedding models"
        models_file = _ensure_embed_models_file()
        models = json.loads(models_file.read_text())
        for model, info in models.items():
            try:
                info["size"] = human_size(pathlib.Path(info["path"]).stat().st_size)
            except FileNotFoundError:
                info["size"] = None
        click.echo(json.dumps(models, indent=2))


class GgufChatModel(llm.Model):
    can_stream = True
    supports_tools = True

    def __init__(
        self,
        model_id,
        model_path,
        n_ctx=0,
        clip_model_path=None,
        chat_handler_class=None,
    ):
        self.model_id = model_id
        self.model_path = model_path
        self.clip_model_path = clip_model_path
        self.chat_handler_class = chat_handler_class
        self.n_ctx = n_ctx  # "0 = from model"
        self._model = None

    def get_model(self):
        if self._model is None:
            if self.chat_handler_class is None:
                # Try without hardcoded chat format to let the model use its native template
                self._model = Llama(
                    model_path=self.model_path,
                    verbose=False,
                    n_ctx=self.n_ctx,
                    # chat_format="chatml-function-calling",  # Let model use its own format
                )
            else:
                chat_handler_class = getattr(llama_chat_format, self.chat_handler_class)
                self._model = Llama(
                    model_path=self.model_path,
                    verbose=False,
                    n_ctx=self.n_ctx,
                    chat_handler_class=chat_handler_class(
                        clip_model_path=self.clip_model_path
                    ),
                )
        return self._model

    def build_messages(self, prompt, conversation):
        """Build messages for the chat template with tool support"""
        messages = []
        current_system = None
        
        # Add conversation history
        if conversation is not None:
            for prev_response in conversation.responses:
                if (
                    prev_response.prompt.system
                    and prev_response.prompt.system != current_system
                ):
                    messages.append(
                        {"role": "system", "content": prev_response.prompt.system}
                    )
                    current_system = prev_response.prompt.system
                messages.append(
                    {"role": "user", "content": prev_response.prompt.prompt}
                )
                messages.append({"role": "assistant", "content": prev_response.text()})
        
        # Add current system message if needed
        if prompt.system and prompt.system != current_system:
            messages.append({"role": "system", "content": prompt.system})
        
        # Add tool results from current prompt if any
        if prompt.tool_results:
            for tool_result in prompt.tool_results:
                messages.append({
                    "role": "tool",  # llama-cpp-python should handle role conversion
                    "name": tool_result.name,
                    "tool_call_id": tool_result.tool_call_id,
                    "content": tool_result.output,
                })
        
        # Add current user message (unless we're just adding tool results)
        if not prompt.tool_results:
            messages.append({"role": "user", "content": prompt.prompt})
        
        return messages

    def _get_format_patterns(self, format_type="generic"):
        """Get regex patterns based on detected format - simplified for gguf"""
        patterns = {
            "llama3x": [
                # Llama 3.x simple format: {"name": "func", "parameters": {...}}
                (r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"parameters"\s*:\s*(\{[^}]*\})\s*\}', 'llama3x'),
            ],
            "generic": [
                # OpenAI format: {"tool_calls": [{"type": "function", "function": {"name": "func", "arguments": {...}}}]}
                (r'\{\s*"tool_calls"\s*:\s*\[\s*\{\s*"type"\s*:\s*"function"\s*,\s*"function"\s*:\s*\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^}]*\})\s*\}\s*\}\s*\]\s*\}', 'openai'),
                # Simple format fallback: {"name": "func", "parameters": {...}}
                (r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"parameters"\s*:\s*(\{[^}]*\})\s*\}', 'simple'),
                # Simple format with arguments: {"name": "func", "arguments": {...}}
                (r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^}]*\})\s*\}', 'simple_args'),
            ]
        }
        return patterns.get(format_type, patterns["generic"])

    def _parse_tool_calls(self, text, available_tools, format_type="generic"):
        """Parse tool calls using format-aware patterns"""
        tool_calls = []
        tool_names = [tool.name for tool in available_tools]
        seen_calls = set()  # Track duplicates
        
        # Get patterns for the detected format
        patterns = self._get_format_patterns(format_type)
        
        for pattern, fmt in patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.MULTILINE)
            
            for match in matches:
                if len(match) >= 2:
                    func_name, args_str = match[0], match[1]
                    
                    # Create a signature to avoid duplicates
                    call_signature = f"{func_name}:{args_str.strip()}"
                    if call_signature in seen_calls:
                        continue
                    
                    if func_name in tool_names:
                        try:
                            # Parse arguments with better error handling
                            if args_str.strip() in ['{}', '']:
                                arguments = {}
                            else:
                                # Handle common JSON parsing issues
                                args_str = args_str.strip()
                                if not args_str.startswith('{'):
                                    args_str = '{' + args_str
                                if not args_str.endswith('}'):
                                    args_str = args_str + '}'
                                arguments = json.loads(args_str)
                            
                            tool_call = llm.ToolCall(
                                name=func_name,
                                arguments=arguments,
                                tool_call_id=f"call_{len(tool_calls)}_{fmt}"
                            )
                            tool_calls.append(tool_call)
                            seen_calls.add(call_signature)
                            
                        except (json.JSONDecodeError, ValueError):
                            continue
        
        return tool_calls

    def execute(self, prompt, stream, response, conversation):
        model = self.get_model()
        messages = self.build_messages(prompt, conversation)
        # Convert tools to the format expected by create_chat_completion
        tools = None
        if prompt.tools:
            tools = []
            for tool in prompt.tools:
                tool_def = {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.input_schema or {"type": "object", "properties": {}}
                    }
                }
                tools.append(tool_def)

        if not stream:
            # Non-streaming execution
            completion = model.create_chat_completion(
                messages=messages,
                tools=tools,
                tool_choice="auto" if tools else None,
            )
            
            choice = completion["choices"][0]
            
            # Handle tool calls if present in native format
            if "tool_calls" in choice["message"] and choice["message"]["tool_calls"]:
                for tool_call in choice["message"]["tool_calls"]:
                    llm_tool_call = llm.ToolCall(
                        name=tool_call["function"]["name"],
                        arguments=json.loads(tool_call["function"]["arguments"]),
                        tool_call_id=tool_call["id"]
                    )
                    response.add_tool_call(llm_tool_call)
                return [choice["message"].get("content", "")]
            else:
                content = choice["message"]["content"]
                # Try parsing tool calls from text if we have tools
                if prompt.tools and content:
                    tool_calls = self._parse_tool_calls(content, prompt.tools)
                    if tool_calls:
                        for tool_call in tool_calls:
                            response.add_tool_call(tool_call)
                return [content]

        # Streaming execution
        completion = model.create_chat_completion(messages=messages, tools=tools, tool_choice="auto" if tools else None, stream=True)
        
        generated_text = ""
        
        for chunk in completion:
            choice = chunk["choices"][0]
            delta = choice.get("delta", {})
            
            # Handle regular content
            delta_content = delta.get("content")
            if delta_content is not None:
                generated_text += delta_content
                yield delta_content
        
        # If we have tools, try parsing the text
        if prompt.tools and generated_text:
            tool_calls = self._parse_tool_calls(generated_text, prompt.tools)
            if tool_calls:
                for tool_call in tool_calls:
                    response.add_tool_call(tool_call)

                # Add some new lines to separate tool calls from the main text
                yield "\n\n"


class GgufEmbeddingModel(llm.EmbeddingModel):
    def __init__(self, model_id, model_path):
        self.model_id = model_id
        self.model_path = model_path
        self._model = None

    def embed_batch(self, texts):
        if self._model is None:
            self._model = Llama(
                model_path=self.model_path, embedding=True, verbose=False
            )
        results = self._model.create_embedding(list(texts))
        return [result["embedding"] for result in results["data"]]


def download_gguf_model(url, models_file_func, aliases):
    """Download a GGUF model and register it in the specified models file"""
    with httpx.stream("GET", url, follow_redirects=True) as response:
        total_size = response.headers.get("content-length")

        filename = url.split("/")[-1]
        download_path = _ensure_models_dir() / filename
        if download_path.exists():
            raise click.ClickException(f"File already exists at {download_path}")

        with open(download_path, "wb") as fp:
            if total_size is not None:
                total_size = int(total_size)
                with click.progressbar(
                    length=total_size,
                    label="Downloading {}".format(human_size(total_size)),
                ) as bar:
                    for data in response.iter_bytes(1024):
                        fp.write(data)
                        bar.update(len(data))
            else:
                for data in response.iter_bytes(1024):
                    fp.write(data)

        click.echo(f"Downloaded model to {download_path}", err=True)
        models_file = models_file_func()
        models = json.loads(models_file.read_text())
        model_id = download_path.stem
        info = {
            "path": str(download_path.resolve()),
            "aliases": aliases,
        }
        models[model_id] = info
        models_file.write_text(json.dumps(models, indent=2))


def human_size(num_bytes):
    """Return a human readable byte size."""
    for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
        if num_bytes < 1024.0:
            break
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} {unit}"
