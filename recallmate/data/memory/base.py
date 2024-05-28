import typing as tp

import pandas as pd
from langchain.prompts import PromptTemplate


def create_prompt_init_default(prompt: PromptTemplate, data: tp.Union[pd.Series, tp.Dict]) -> str:
    """
    Deafualt function that creates a prompt based on the input template and data row.

    Parameters:
        prompt (PromptTemplate): The template for the prompt.
        data_row (pd.Series | Dict): The data row to extract variables for the prompt.

    Returns:
        str: The created prompt.
    """
    prompt_init = prompt.format(
        **{
            variable: data[variable] for variable in prompt.input_variables
        }
    )
    return prompt_init


class BaseMemory:
    """
    Base class for memory.

    Parameters:
        _data (pd.DataFrame): The dataframe with meta-information. 
        col_id (Any): The column with id. Defaults to "id".
        _memory (Dict): The memory.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        *args: tp.Any,
        col_id: tp.Any = "id",
        **kwargs: tp.Any,
    ) -> None:
        self._data = data.copy()
        self.col_id = col_id

        self._data.drop_duplicates(subset=col_id, keep="first", inplace=True, ignore_index=True)
        
        self._memory: tp.Dict[tp.Any, tp.Any] = {}

    def _create_default_prompt(self) -> str:
        """
        Create a default prompt based on the columns of the data and the them columns.

        Returns:
            str: The default prompt string.
        """
        default_prompt = PromptTemplate.from_template(
            " ".join([f"{col}: " + "{" + str(col) + "};" for col in self._data.columns if col != self.col_id])
        )
        return default_prompt


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
        """
        Create memory.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self._data[self.col_id].unique())

    def __getitem__(self, idx: tp.Any) -> tp.Any:
        if idx in self._memory:
            return self._memory[idx]
        else:
            raise KeyError(f"Not find id {idx} in memory.")
    
    def update_memory(self, idx: tp.Any, new_memory: tp.Any, *args: tp.Any, **kwargs: tp.Any) -> None:
        """
        Update the memory with a new value for a given item_id.

        Parameters:
            idx (Any): The identifier of the element in memory to be updated.
            new_memory (Any): The new value to be stored in memory for the given element.
        
        Returns:
            None
        """
        if idx in self._memory:
            self._memory[idx] = new_memory
        else:
            raise KeyError(f"Not find id {idx} in memory.")
    
    @property
    def get_ids(self) -> tp.List[tp.Any]:
        """
        Return a list of ids in the memory.

        Returns:
            tp.List[tp.Any]: List of ids are held in memory.
        """
        return self._data[self.col_id].unique().tolist()
    
    @property
    def get_memory(self) -> tp.Dict[tp.Any, tp.Any]:
        """
        Return memory.

        Returns:
            tp.Dict[tp.Any, tp.Any]: The memory.
        """
        return self._memory
    
    @property
    def get_data(self) -> pd.DataFrame:
        """
        Return input data.

        Returns:
            pd.DataFrame: Input data.
        """
        return self._data
