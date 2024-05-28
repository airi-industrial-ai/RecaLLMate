import typing as tp

import pandas as pd
from langchain.prompts import PromptTemplate

from recallmate.columns import Columns

from .base import BaseMemory, create_prompt_init_default
from .item_memory import ItemMemory


class UserMemory(BaseMemory):
    """
    User memory constructor.

    Parameters:
        data (pd.DataFrame): The dataframe containing the user data. 
        interactions (pd.DataFrame): The dataframe containing the interactions of the users.
        item_memory (ItemMemort): The item memory.
        use_prompt_init (bool): Use information of the user or no. Defults True.
        col_id (str): The column name of the user id. Defaults to "user_id".
        prompt_init (PromptTemplate): The prompt template for user's information. Defaults None.
        create_prompt (Callable): Function for creating prompt with user's information.
        user_long_memory_prompt_init (PromptTemplate): The prompt template for the user long memory per item. 
            Must includes one variable `history`. Defaults PromptTemplate.from_template(
            "I interected with contents to the following films (in historical order): {history}"
        )
        user_short_memory_prompt_init (PromptTemplate): The prompt template for the user short memory. Defaults None.
        short_memory_create_prompt (Callable): Function for creating short-term user's prompt. Defaults None.
        long_memory_n (int): The number of items to keep in the long term memory. Defaults to 20.
        short_memory_n (int): The number of items to keep in the short term memory. Defaults to 5.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        interactions: pd.DataFrame,
        item_memory: ItemMemory,
        use_prompt_init: bool = True,
        col_id: str = Columns.User,
        prompt_init: tp.Optional[PromptTemplate] = None,
        create_prompt: tp.Callable = create_prompt_init_default,
        user_long_memory_prompt_init: PromptTemplate = PromptTemplate.from_template(
            "I interected with contents to the following films (in historical order): {history}"
        ),
        user_short_memory_prompt_init: tp.Optional[PromptTemplate] = None,
        short_memory_create_prompt: tp.Optional[PromptTemplate] = None,
        long_memory_n: int = 20,
        short_memory_n: int = 5,
    ) -> None:
        super().__init__(data=data, col_id=col_id)

        self._memory["long-term"] = {}
        self._memory["short-term"] = {}
        
        self.item_memory = item_memory
        self.item_data = self.item_memory.get_data
        self.col_item_id = self.item_memory.col_id

        self.use_prompt_init = use_prompt_init

        interactions = interactions.groupby(col_id, as_index=False)[self.col_item_id].agg(list)
        self.history_per_users = dict(zip(interactions[col_id], interactions[self.col_item_id]))

        if self.use_prompt_init:
            self.prompt_init = prompt_init
            if self.prompt_init is None:
                self.prompt_init = self._create_default_prompt()
            self.create_prompt = create_prompt
            self._check_input_variables(self.prompt_init)

        # Long-term memory
        if (
            len(user_long_memory_prompt_init.input_variables) != 1 
            and user_long_memory_prompt_init.input_variables[0] != "history"
        ):
            raise KeyError(
                "`user_long_memory_per_item_prompt_init` must only have one parameter, and that's `{history}`."
            )
        self.user_long_memory_prompt_init = user_long_memory_prompt_init
        self.long_memory_n = long_memory_n
        
        # Short-term memory
        self.user_short_memory_prompt_init = user_short_memory_prompt_init
        self.short_memory_create_prompt = short_memory_create_prompt
        self.short_memory_n = short_memory_n

        if self.long_memory_n < self.short_memory_n:
            raise ValueError(
                "`long_memory_n` must be greater than `short_memory_n`."
            )
    
        if self.user_short_memory_prompt_init is not None:
            if self.short_memory_create_prompt is None:
                raise ValueError(
                    "If `user_short_memory_prompt_init` is not None, the `short_memory_create_prompt` must be not None."
                )
            self._check_input_variables(self.user_short_memory_prompt_init, self.item_data)

        self._create_memory()
        
    def _check_input_variables(self, prompt: PromptTemplate, data: tp.Optional[pd.DataFrame] = None) -> None:
        """
        Checks if all variables in `prompt_init` are columns in the dataframe.
        """
        if data is not None:
            columns = data.columns
        else:
            columns = self._data.columns
            
        if prompt is not None:
            for variable in prompt.input_variables:
                if variable not in columns:
                    raise KeyError(f"Variable {variable} from `prompt_init` is not in the columns of the dataframe.")

    def _create_memory(self) -> None:
        for user_id in self.history_per_users:
            user_history = self.history_per_users[user_id]
            
            # Long-term memory
            user_history_long = user_history[-self.long_memory_n :]
                
            long_memory = []
            for item_id in user_history_long:
                try:
                    long_memory.append(self.item_memory[item_id])
                except:
                    print(f"[WARNING]: Don't find item id: {item_id}")
                    continue
            
            # Short-term memory
            short_memory = ""
            if self.user_short_memory_prompt_init is not None:
                user_history_short = user_history[-self.short_memory_n :]

                values_for_create_prompt = {}
                for variable in self.user_short_memory_prompt_init.input_variables:
                    values_for_create_prompt[variable] = str(
                        self.item_data[
                            self.item_data[self.col_item_id].isin(user_history_short)
                        ][variable].unique()
                    )
                short_memory = self.short_memory_create_prompt(self.user_short_memory_prompt_init, values_for_create_prompt)

            user_overview = ""
            if self.use_prompt_init:
                user_overview = self.create_prompt(self.prompt_init, self._data[self._data[self.col_id] == user_id].iloc[0])

            self._memory["long-term"][user_id] = long_memory
            self._memory["short-term"][user_id] = (user_overview + " " + short_memory).strip()

    def _create_long_term_prompt(self, idx: tp.Any) -> str:
        long_memory = " ".join([f"{idx + 1}. {item}" for idx, item in enumerate(self._memory["long-term"][idx])])
        long_memory = self.user_long_memory_prompt_init.format(history=long_memory)
        return long_memory
    
    def __getitem__(self, idx: tp.Any) -> tp.Any:
        if idx in self._memory["long-term"]:
            return {
                "short-term": self._memory["short-term"][idx],
                "long-term": self._create_long_term_prompt(idx)
            }
        else:
            raise KeyError(f"Not find id {idx} in memory.")
    
    def update_memory(self, idx: int, new_memory: tp.Any = None, new_short_memory: str = None) -> str:
        if new_memory in self.item_memory.get_ids:
            self._memory["long-term"][idx].append(self.item_memory[new_memory])
            self.history_per_users[idx].append(new_memory)
        else:
            raise KeyError(f"Don't know User ID {idx}.")
                
        if new_short_memory is not None:
            self._memory["short-term"][idx] = new_short_memory

    @property
    def get_long_term_memory(self) -> tp.Dict[int, str]:
        """
        Returns the short-term memory
        
        Returns:
            tp.List[tp.Any]: A list of keys in the short-term memory.
        """
        lont_term_memory = {
            user_id: self._create_long_term_prompt(user_id)
            for user_id in self._memory["long-term"]
        }
        return lont_term_memory

    @property
    def get_short_term_memory(self) -> tp.Dict[int, str]:
        """
        Returns the short-term memory
        
        Returns:
            tp.List[tp.Any]: A list of keys in the short-term memory.
        """
        return  self._memory["short-term"]

    @property
    def get_history_per_users(self) -> tp.Dict[int, tp.List[int]]:
        """
        Returns the history per user.

        Returns:
            tp.Dict[int, tp.List[int]]: History per users.
        """
        return self.history_per_users
