import typing as tp

from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from recallmate.columns import Columns
from recallmate.data.memory import ItemMemory
from recallmate.llm.open_source_llm_embedding import OpenSourceLLMEmbeddings

from .base import RetrievalRecommenderBase


class RetrievalRecommenderSimple(RetrievalRecommenderBase):
    """
    Simple retireval recommender.

    Parameters:
        item_memory (ItemMemory): Memory of the items.
        col_item_id (str): Column name with item id.
        text_splitter (TextSplitter): Splitter for documents.
        docs (List[Documents]): List of documents for the retrieval.
        embeddings_model (OpenAIEmbeddings | HuggingFaceEmbedding | OpenSourceLLMEmbeddings): Embedding model.
        vectorstore (VectorStore): Document store.
        retriever (VectorStoreRetriever): Documnet retriever.
    """

    def __init__(
        self,
        item_memory: ItemMemory,
        embeddings_model: tp.Union[OpenAIEmbeddings, HuggingFaceEmbeddings, OpenSourceLLMEmbeddings],
        col_item_id: str = Columns.Item,
        text_splitter_args: tp.Dict[str, tp.Any] = dict(
            chunk_size=1000, chunk_overlap=0
        ),
        log: bool = True,
    ) -> None:
        super().__init__(
            item_memory=item_memory,
            embeddings_model=embeddings_model,
            col_item_id=col_item_id,
            text_splitter_args=text_splitter_args,
            log=log,
        )

    def _create_vectore_store(self) -> FAISS:
        try:
            return FAISS.from_documents(self.docs, self.embeddings_model)
        except Exception as e:
            raise ValueError(f"OpenAI API key is invalid. Message: {e}")
