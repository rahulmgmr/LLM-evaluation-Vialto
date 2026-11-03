"""Optional Langfuse observability for evaluation runs."""

import os
import atexit
from contextlib import nullcontext


# Step 1: Enable tracing only when a complete Langfuse credential pair exists.
def setup_langfuse():
    """Initialize Langfuse and its LangChain callback when credentials exist."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    if not public_key and not secret_key:
        return None, None
    if not public_key or not secret_key:
        raise ValueError(
            "Set both LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY to enable tracing."
        )

    try:
        from langfuse import get_client
        from langfuse.langchain import CallbackHandler
    except ImportError as exc:
        raise RuntimeError(
            "Langfuse tracing is configured but dependencies are missing. "
            "Install requirements.txt."
        ) from exc

    client = get_client()
    atexit.register(client.flush)
    return client, CallbackHandler()


# Step 2: Let callers trace work or use a no-op context when tracing is off.
def observation(client, *, name, as_type="span", **kwargs):
    """Return a no-op context when Langfuse is not configured."""
    if client is None:
        return nullcontext(None)
    return client.start_as_current_observation(name=name, as_type=as_type, **kwargs)


# Step 3: Send any buffered observations before the evaluation process exits.
def flush_langfuse(client):
    if client is not None:
        client.flush()
