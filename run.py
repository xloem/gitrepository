#!/usr/bin/env python3
import torch, transformers
from langchain import HuggingFacePipeline, ConversationChain, LLMChain, PromptTemplate
from langchain.chains.conversation.memory import ConversationalBufferWindowMemory


template = """Assistant is a large language model trained by OpenAI.

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

{history}
Human: {human_input}
Assistant:"""

prompt = PromptTemplate(
    input_variables=["history", "human_input"], 
    template=template
)


#MODEL='HuggingFaceH4/opt-iml-max-30b'
MODEL, TASK ='facebook/opt-iml-max-30b', 'text-generation'
chatgpt_chain = LLMChain(
    llm=HuggingFacePipeline.from_model_id(
        MODEL, TASK,
        model_kwargs=dict(
            temperature=0,
            max_length=2048,
#            begin_suppress_tokens=[
#                transformers.AutoTokenizer
#                    .from_pretrained(MODEL)
#                    .encode(
#                        "I'm sorry, I can't answer that question.",
#                        add_special_tokens=False
#                    )[0]
#            ],
            device_map="auto",
            offload_folder="offload",
            torch_dtype=torch.float16)), 
    prompt=prompt, 
    verbose=True, 
    memory=ConversationalBufferWindowMemory(k=2),
)

while True:
    print(chatgpt_chain.predict(human_input=input("> ")))
