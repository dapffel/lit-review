import faiss
import litellm
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter


class VectorMemory:
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.model = model
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.texts: list[str] = []
        self.index: faiss.IndexFlatL2 | None = None

    async def _embed(self, texts: list[str]) -> np.ndarray:
        response = await litellm.aembedding(model=self.model, input=texts)
        return np.array([d["embedding"] for d in response.data], dtype=np.float32)

    async def add(self, documents: list[str]):
        chunks = []
        for doc in documents:
            chunks.extend(self.splitter.split_text(doc))
        if not chunks:
            return

        embeddings = await self._embed(chunks)
        self.texts.extend(chunks)

        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    async def query(self, text: str, k: int = 3) -> list[str]:
        if not self.index or self.index.ntotal == 0:
            return []
        embedding = await self._embed([text])
        _, indices = self.index.search(embedding, min(k, self.index.ntotal))
        return [self.texts[i] for i in indices[0] if i < len(self.texts)]
