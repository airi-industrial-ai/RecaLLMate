import typing as tp

import numpy as np
import pandas as pd
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from tqdm import tqdm

from recallmate.columns import Columns
from recallmate.data.memory.base import BaseMemory
from recallmate.llm.open_source_llm_embedding import OpenSourceLLMEmbeddings


class RetrievalRecommenderBase:
    """
    Recommender that uses a retrieval-based approach to recommend similar content based on user interactions.

    Parameters:
        log (bool): Log retrieval.
        col_item_id (str): Column name with item id.
        item_memory (ItemMemory): Memory of the items.
        text_splitter (TextSplitter): Splitter for documents.
        docs (List[Documents]): List of documents for the retrieval.
        embeddings_model (OpenAIEmbeddings | HuggingFaceEmbedding | OpenSourceLLMEmbeddings): Embedding model.
        vectorstore (VectorStore): Document store.
        retriever (VectorStoreRetriever): Documnet retriever.
    """

    def __init__(
        self,
        item_memory: BaseMemory,
        embeddings_model: tp.Union[OpenAIEmbeddings, HuggingFaceEmbeddings, OpenSourceLLMEmbeddings],
        *args: tp.Any,
        col_item_id: str = Columns.Item,
        text_splitter_args: tp.Dict[str, tp.Any] = dict(
            chunk_size=1000, chunk_overlap=0
        ),
        log: bool = True,
        **kwargs: tp.Any,
    ) -> None:
        self.log = log
        
        self.col_item_id = col_item_id
        self.item_memory = item_memory
        self.user_memory: BaseMemory

        # Create documents (preprocess)
        self.text_splitter = CharacterTextSplitter(**text_splitter_args)
        self.docs = self._prepare_data()
        
        self.embeddings_model = embeddings_model

        self.vectorstore = self._create_vectore_store()
        self.retriever: VectorStoreRetriever

    def _prepare_data(self) -> tp.List[Document]:
        documents = self.text_splitter.create_documents(
            texts=list(self.item_memory.get_memory.values()),
            metadatas=list({self.col_item_id: _id} for _id in self.item_memory.get_memory.keys())
        )
        return documents

    def _create_vectore_store(self) -> VectorStore:
        raise NotImplementedError

    def _recommend_per_user(self, query: str) -> tp.List[int]:
        documents = self.retriever.get_relevant_documents(query, verbose=self.log)
        reco_item_ids = self.parse(documents)
        return reco_item_ids

    def parse(self, docs: tp.List[Document]) -> tp.List[tp.Any]:
        item_ids = [doc.metadata[self.col_item_id] for doc in docs]
        return item_ids

    def update_item_memory(self, item_memory: BaseMemory) -> None:
        """
        Update the item memory with a new value and prepare the data and vector store accordingly.

        Parameters:
            item_memory (BaseMemory): The new item memory to be updated.
        
        Returns:
            None
        """
        self.item_memory = item_memory
        self.docs = self._prepare_data()
        self.vectorstore = self._create_vectore_store()

    def run(
        self,
        user_memory: BaseMemory,
        *args: tp.Any,
        instruct: str = None,
        users_for_recommend: tp.Sequence[tp.Any] = None,
        use_user_memory_short: bool = True,
        use_user_memory_long: bool = True,
        search_type: str = "similarity",
        search_kwargs: tp.Dict[str, tp.Any] = {"k": 10},
        add_rank: bool = True,
        **kwargs: tp.Any
    ) -> pd.DataFrame:
        """
        Runs the recommendation process for a list of users based on their memory and instructions.
        
        Parameters:
            user_memory (BaseMemory): The memory of the users.
            instruct (str, optional): Additional instructions for the recommendation process. Defaults to None.
            users_for_recommend (tp.Sequence[tp.Any], optional): Specific users for whom recommendations should be generated. Defaults to None.
            use_user_memory_short (bool): Flag to indicate whether to use short-term memory in the recommendation process. Defaults to True.
            use_user_memory_long (bool): Flag to indicate whether to use long-term memory in the recommendation process. Defaults to True.
            search_type (str): The type of search to be performed. Defaults to "similarity".
            search_kwargs (tp.Dict[str, tp.Any]): Keyword arguments for the search process. Defaults to {"k": 10}.
            add_rank (bool): Flag to add ranking to the recommendations. Defaults to True.
        
        Returns:
            pd.DataFrame: A DataFrame containing user recommendations with user IDs, item IDs, and ranks (optional.
        """
        self.user_memory = user_memory
        self.retriever = self.vectorstore.as_retriever(
            search_type=search_type, search_kwargs=search_kwargs
        )
        
        users_ids = self.user_memory.get_ids
        if users_for_recommend is not None:
            users_ids = [user_id for user_id in users_for_recommend if user_id in users_ids]

        user_ids, reco_ids = [], []
        for user_id in tqdm(users_ids):
            user_query = ""
            if instruct is not None:
                user_query += f"{instruct} "
            if use_user_memory_short:
                user_query += f"{user_memory[user_id]['short-term']} "
            if use_user_memory_long != -1:
                user_query += user_memory[user_id]["long-term"]
            user_query = user_query.strip()
            
            reco_item_ids = self._recommend_per_user(user_query)

            user_ids.extend(np.repeat(user_id, len(reco_item_ids)))
            reco_ids.extend(reco_item_ids)

        reco = pd.DataFrame(
            {
                Columns.User: user_ids,
                Columns.Item: reco_ids,
            }
        )

        if add_rank:
            reco[Columns.Rank] = reco.groupby(Columns.User).cumcount() + 1

        return reco