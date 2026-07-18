from fashion_retrieval.indexer.attribute_extraction import extract_attributes_from_text
from fashion_retrieval.indexer.config import IndexConfig
from fashion_retrieval.indexer.models import BlipFashionEncoder


class QueryParser:
    def __init__(self, config: IndexConfig) -> None:
        self.encoder = BlipFashionEncoder(config.blip_model_name, config.device)

    def parse(self, query: str) -> dict:
        return {
            "text": query,
            "embedding": self.encoder.encode_texts([query])[0],
            "attributes": extract_attributes_from_text(query),
        }

