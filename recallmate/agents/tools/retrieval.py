import typing as tp

from langchain.tools import StructuredTool
from langchain_core.pydantic_v1 import BaseModel as PydanticBaseModel
from langchain_core.pydantic_v1 import Field

from recallmate.agents.tools import create_tool
from recallmate.data.memory.base import BaseMemory
from recallmate.tasks.information_retrieval.base import RetrievalRecommenderBase


class BaseModel(PydanticBaseModel):
    class Config:
        arbitrary_types_allowed = True


class RetrieveInput(BaseModel):
    """Input to the task."""

    user_memory: BaseMemory = Field(
        description="Data structure for working with user memory"
    )
    instruct: tp.Optional[str] = Field(
        description="Additional information to provide guidance",
        default=None,
    )
    users_for_recommend: tp.Sequence[tp.Any] = Field(
        description="Specific users for whom recommendations should be generated",
        default=[],
    )
    use_user_memory_short: bool = Field(
        description=(
            "Flag indicating whether users' short-term memory should be taken "
            "into account when searching for relevant recommendations"
        ),
        default=True,
    )
    use_user_memory_long: bool = Field(
        description=(
            "Flag indicating whether users' long-term memory should be taken "
            "into account when searching for relevant recommendations"
        ),
        default=True,
    )
    search_type: str = Field(
        description="The type of search that is used to provide recommendations",
        default="similarity",
    )
    search_kwargs: tp.Dict[str, tp.Any] = Field(
        description="Keyword arguments for the search process",
        default={"k": 10},
    )
    add_rank: bool = Field(
        description="Flag to add a rating to the output for recommendations",
        default=True,
    )


def create_retrieval_tool(
    retrieval: RetrievalRecommenderBase,
    name: str = "retrieval_recommender",
    description: str = (
        "Search content that is most similar to content from previous interactions with the recommender system.\n"
        "If you have any questions about searching related content, you should use this tool!\n"
    ),
    args_schema: tp.Optional[BaseModel] = RetrieveInput,
    return_direct: bool = False,
    infer_schema: bool = True,
) -> StructuredTool:
    """Create a tool to do retrieval recommender.

    Parameters:
        retrieval (Recommender): The retrieval to use for the retrieval
        name (str): The name for the tool. This will be passed to the language model,
            so should be unique and somewhat descriptive. Default  "retrieval_recommender".
        description (str): The description for the tool. This will be passed to the language
            model, so should be descriptive. Default (
                "Search content that is most similar to content from previous interactions with the recommender system.\n"
                "If you have any questions about searching related content, you should use this tool!\n"
            )
        args_schema (BaseModel): The schema of the tool's input arguments. Default RetrieveInput.
        return_direct (bool): Whether to return the result directly or as a callback.
        infer_schema (bool): Whether to infer the schema from the function's signature.

    Returns:
        (Tool): Tool class to pass to an agent
    """
    return create_tool(
        task=retrieval,
        name=name,
        description=description,
        args_schema=args_schema,
        return_direct=return_direct,
        infer_schema=infer_schema,
    )
