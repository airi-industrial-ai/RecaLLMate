# RecaLLMate
RecaLLMate: python framework for flexible and reproducible study of large language models based recommender systems
![Scheme](docs/RecaLLMate.jpg)

The framework includes the following key modules: 
- working with user and item features (data)
- functionality for loading and fine-tuning LLMs (llm)
- solving recommendation problems with LLMs (tasks)
- interface for wrapping and using classical RecSys modules
- functionality for using agent-based approaches, generating instruments, memory and parsing responses for agents (agents)
- evaluation of RecSys results (evaluate)

# Overview
Language models (LLMs) and multi-modal models excel at encoding content for recommender systems. 
This encoding facilitates item indexing for swift candidate retrieval and results re-ranking using LLMs. 
Additionally, LLMs can seamlessly incorporate recommendations into messages during human communication. 
There is also a large amount of research coming out at the moment in the area of applying LLMs to recommendation system tasks. 
However, the code base of these studies is very diverse and complex to reproduce, and existing libraries for building recommender systems are not well suited for applying LLMs.
This limitation leads to the fact that LLMs for recommender systems exist separately from classical recommendation approaches. 
In addition, the lack of a unified tool prevents comparing different models and methods. 
It creates uncertainty in the research results, making it challenging to apply the investigated approaches to industrialized problems.

# Examples
First of all, you need to install the necessary dependencies:
```
pip install -r requirements/requirements.txt
```

 - [information retrieval](./examples/information_retrieval.ipynb)
