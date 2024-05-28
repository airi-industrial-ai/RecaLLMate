import typing as tp

import pandas as pd
from langchain.prompts import PromptTemplate

from recallmate.columns import Columns

from .base import BaseMemory, create_prompt_init_default


class ItemMemory(BaseMemory):
    """
    Constructor for item memory.

    Parameters:
        data (pd.DataFrame): The dataframe containing the item data. 
        col_id (str): The column name of the item id. Defaults to "item_id".
        prompt_init (PromptTemplate): The promt for initialization.
        create_prompt (Callable): The function for translating information into a string.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        col_id: str = Columns.Item,
        prompt_init: tp.Optional[PromptTemplate] = None,
        create_prompt: tp.Callable = create_prompt_init_default,
    ) -> None:
        super().__init__(data=data, col_id=col_id)

        if prompt_init is not None:
            self.prompt_init = prompt_init
        else:
            self.prompt_init = self._create_default_prompt()
            
        self.create_prompt = create_prompt
        
        self._check_input_variables(self.prompt_init)
        self._create_memory()
    
    def _create_memory(self) -> None:
        self._memory = {
            self._data.iloc[row_idx][self.col_id]: self.create_prompt(self.prompt_init, self._data.iloc[row_idx])
            for row_idx in range(len(self._data)) 
        }
