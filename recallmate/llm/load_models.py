import typing as tp

from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI, OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

from .open_source_llm_embedding import OpenSourceLLMEmbeddings


def load_model_openai(
    model_name: str,
    openai_api_key: str,
    mode: str = "generation",
    model_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
) -> tp.Union[OpenAI, ChatOpenAI, OpenAIEmbeddings]:
    """
    Load an OpenAI embedding model using the specified model name and API key.
    
    Parameters:
        model_name (str): The name of the model to load.
        openai_api_key (str): The API key for accessing the OpenAI service.
        mode (str): Type of the model for loading. Default is `generation`.
        embeddings_model_args (Optional[Dict[str, Any]]): Additional arguments for the embeddings model. Default is None.
        
    Returns:
        (OpenAI | ChatOpenAI | OpenAIEmbeddings): An instance of the OpenAIEmbeddings model loaded with the specified parameters.
    """
    if mode == "generation":
        model = OpenAI(model=model_name, openai_api_key=openai_api_key, **model_kwargs)
    elif mode == "chat":
        model = ChatOpenAI(model=model_name, openai_api_key=openai_api_key, **model_kwargs)
    elif mode == "embedding":
        model = OpenAIEmbeddings(openai_api_key=openai_api_key, model=model_name)
    else:
        raise ValueError("Parameter `mode` must be only [`generation`, `chat`, `embedding`].")
    return model


def load_model_open_source(
    model_name: str,
    device_id: tp.Union[int, str] = -1,
    hf_tokens: str = None,
    mode: str = "generation",
    bits_and_bits_params: tp.Optional[tp.Dict[str, tp.Any]] = None,
    generation_params: tp.Dict[str, tp.Any] = None,
    generation_inference_params: tp.Dict[str, tp.Any] = None,
    st_multi_process: bool = False,
    st_model_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
    st_encode_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
    emb_model_context_length: int = 512,
    emb_model_batch_size: int = 1,
) -> tp.Union[HuggingFaceEmbeddings, HuggingFacePipeline, Embeddings]:
    """
    Generate a HuggingFace pipeline for text generation using the specified model and parameters.

    Parameters:
        model_name (str): The name of the pre-trained model to load
        device_id (int | str): Device for inference. Defaults to -1.
        hf_tokens (str): The API tokens for HuggingFace.
        mode (str): Type of the model for loading. Defaults to `generation`.
        bits_and_bits_params (Dict[str, Any]): Additional parameters for the BitsAndBytesConfig. Optional.
            Used for `mode` is `"generation"`.
        generation_params (Dict[str, Any]): Additional parameters for text generation. Defaults to None.
            Used for `mode` is `"generation"`.
        generation_inference_params (Dict[str, Any]): Additional parameters for inference text generation. Optional. 
            Used for `mode` is `"generation"`.
        st_multi_process (bool): Whether to use multiple processes for loading the model. Defaults to False.
            Used for `mode` is `"sentence_transformers"`.
        st_model_kwargs (Optional[Dict[str, Any]]): Additional keyword arguments for the model initialization. Defaults to None.
            Used for `mode` is `"sentence_transformers"`.
        st_encode_kwargs (Optional[Dict[str, Any]]): Additional keyword arguments for encoding text. Defaults to None.
            Used for `mode` is `"sentence_transformers"`.
        emb_model_context_length (int): Length context window for embedding model. Defaults to 512.
            Used for `mode` is `"embedding"`.
        emb_model_batch_size (int): Batch size for embedding model. Defaults to 1.
            Used for `mode` is `"embedding"`.

    Returns:
        HuggingFacePipeline: A pipeline for text generation using the loaded model.
    """
    if mode == "sentence_transformers":
        model = HuggingFaceEmbeddings(
            model_name=model_name,
            multi_process=st_multi_process, 
            model_kwargs=st_model_kwargs, 
            encode_kwargs=st_encode_kwargs
        )
    elif mode == "generation" or mode == "embedding":
        if bits_and_bits_params is not None:
            bnb_config = BitsAndBytesConfig(**bits_and_bits_params)
            model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_tokens, device_map="auto", quantization_config=bnb_config)
        else:
            if device_id != -1 and device_id != "auto":
                model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_tokens).to(f"cuda:{device_id}")
            elif device_id == "auto":
                model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_tokens, device_map="auto")
            else:
                model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_tokens)

        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_tokens)

        if mode == "generation":
            text_generation_pipeline = pipeline(
                model=model, tokenizer=tokenizer, task="text-generation", **generation_params
            )
            model = HuggingFacePipeline(pipeline=text_generation_pipeline, **generation_inference_params)
        elif mode == "embedding":
            model = OpenSourceLLMEmbeddings(
                model=model, 
                tokenizer=tokenizer,
                batch_size=emb_model_batch_size,
                context_length=emb_model_context_length)
    else:
        raise ValueError("Parameter `mode` must be only [`generation`, `sentence_transformers`, `embedding`].")

    return model
