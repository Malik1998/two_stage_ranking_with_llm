import time
import numpy as np
from openai import OpenAI


class LLMEmbedder:
    def __init__(
        self,
        api_key: str,
        model: str = "qwen/qwen3-embedding-8b",
        batch_size: int = 16,
        sleep_sec: float = 0.5,
    ):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model = model
        self.batch_size = batch_size
        self.sleep_sec = sleep_sec

    def embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            resp = self.client.embeddings.create(
                model=self.model,
                input=batch,
                encoding_format="float",
            )

            embeddings.extend(
                [np.array(d.embedding, dtype=np.float32) for d in resp.data]
            )

            time.sleep(self.sleep_sec)

        return embeddings
