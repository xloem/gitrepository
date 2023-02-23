#!/usr/bin/env python3
import asyncio, os
import torch, transformers
import langchain
from langchain import ConversationChain, LLMChain

langchain.llm_cache = langchain.cache.SQLiteCache(database_path="langchain.db")

#MODEL, TASK ='HuggingFaceH4/opt-iml-max-30b', 'text-generation'
MODEL, TASK ='facebook/opt-iml-max-30b', 'text-generation'
#MODEL, TASK ='facebook/opt-iml-max-1.3b', 'text-generation'
#MODEL, TASK ='google/flan-t5-xxl', 'text2text-generation'
#MODEL, TASK ='google/flan-t5-xl', 'text2text-generation'

# maybe:
# - look at how memory is hooked in and parameterize it
# - look at how tools are generated and add some
#     it might be nice to have it review execution steps for danger
# - ensure saving and loading include memory content
# - customize the prompt to simplify the below




#template = """Assistant is a large language model trained by OpenAI.
#
#Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.
#
#Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.
#
#Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.
#
#{history}
#Human: {input}
#Assistant:"""
#
#prompt = langchain.PromptTemplate(
#    input_variables=["history", "input"], 
#    template=template
#)


begin_suppress_tokens=[
    transformers.AutoTokenizer
        .from_pretrained(MODEL)
        .encode(
            "I'm sorry, I can't answer that question.",
            add_special_tokens=False
        )[0],
    transformers.AutoTokenizer
        .from_pretrained(MODEL)
        .encode(
            "[your response here]",
            add_special_tokens=False
        )[0],
]

class Agent:
    def __init__(self, MODEL, TASK, **kwparams):
        self.path = MODEL + '.agent.json'
        self.modelname = MODEL
        self.taskname = TASK
        self.kwparams = kwparams
    def __enter__(self):
        llm = langchain.HuggingFacePipeline.from_model_id(
            self.modelname, self.taskname,
            model_kwargs=dict(
                temperature=0,
                do_sample=False,
                max_length=2048,
                device_map="auto",
                offload_folder="offload",
                torch_dtype="auto",#torch.float16
                begin_suppress_tokens=begin_suppress_tokens,
            )
        )
        tools = langchain.agents.load_tools([], llm=llm)
        memory = langchain.chains.conversation.memory.ConversationalBufferWindowMemory(memory_key='chat_history') #ConversationBufferMemory
        self.agent = langchain.agents.ConversationalAgent.from_llm_and_tools(
            llm,
            tools,
            memory=memory,
            **self.kwparams
            #prompt=prompt, 
            #verbose=True, 
            #memory=ConversationalBufferWindowMemory(k=2),
        )
        self.executor = langchain.agents.AgentExecutor(
            agent=self.agent,
            tools=tools,
            memory=memory,
            **self.kwparams
            #verbose=True, 
        )
        return self.executor
    def __exit__(self, *params):
        #self.executor.agent.save(self.path)
        del self.executor
        del self.agent

async def main():
    with Agent(MODEL, TASK, verbose=True) as executor:
        while True:
            #print(await executor.arun(input=input("> ")))
            print(executor.run(input=input("> ")))

if __name__ == '__main__':
    asyncio.run(main())
