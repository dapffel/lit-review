import os

# Disable LangSmith/LangGraph tracing during tests so runs make no network calls and
# don't spawn the LangSmith background thread. Individual tracing tests set their own env.
# setdefault (plus python-dotenv's non-overriding load) keeps any explicit env the user set.
os.environ.setdefault("LANGSMITH_TRACING", "false")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
