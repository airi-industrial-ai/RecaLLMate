import typing as tp

from langchain.tools import StructuredTool
from langchain_core.pydantic_v1 import BaseModel


def create_tool(
    task,
    name: str,
    description: str,
    args_schema: BaseModel,
    return_direct: bool = False,
    infer_schema: bool = True,
    **kwargs: tp.Any,
) -> StructuredTool:
    """Create a tool to do any recommendation task.

    Args:
        task (Recommender): The task to use for the retrieval
        name (str): The name for the tool. This will be passed to the language model,
            so should be unique and somewhat descriptive.
        description (str): The description for the tool. This will be passed to the language
            model, so should be descriptive.
        args_schema (BaseModel): The schema of the tool's input arguments.
        return_direct (bool): Whether to return the result directly or as a callback.
        infer_schema (bool): Whether to infer the schema from the function's signature.

    Returns:
        (Tool): Tool class to pass to an agent
    """

    try:
        return StructuredTool.from_function(
            func=task.run,
            name=name,
            description=description,
            args_schema=args_schema,
            return_direct=return_direct,
            infer_schema=infer_schema,
            **kwargs,
        )
    except Exception as e:
        raise ValueError(f"Error creating tool: {e}")
