# pip install chainlit langchain_google_genai langchain langchain_community sympy replicate
import chainlit as cl
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import (
    HarmBlockThreshold,
    HarmCategory,
)
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.utilities import PythonREPL
from langchain.agents import Tool, initialize_agent, AgentType
from dotenv import load_dotenv
import os
import replicate
import pandas
import sklearn

load_dotenv()

class ReplicateImageTool:
    def __init__(self, api_token):
        self.api_token = api_token
        self.client = replicate.Client("r8_8xMW7QSdki4vZ5Rn70RF5CJjWPXvaot2gBYRo")

    def generate_image(self, prompt: str):
        input = {
            "width": 768,
            "height": 768,
            "prompt": prompt,
            "refine": "expert_ensemble_refiner",
            "apply_watermark": False,
            "num_inference_steps": 25
        }
        output = self.client.run(
            "stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
            input=input
        )
        return output

@cl.on_chat_start
def aelm():
    os.environ["GOOGLE_API_KEY"] = "AIzaSyA3dePIEUpXm-pDqDKqmKDPxkW1stqBXqY"
    os.environ["REPLICATE_API_TOKEN"] = "r8_8xMW7QSdki4vZ5Rn70RF5CJjWPXvaot2gBYRo"

    if os.getenv("GOOGLE_API_KEY") is None or os.getenv("REPLICATE_API_TOKEN") is None:
        raise ValueError("Mundus Announcement: AELM-Aristotle Upgrade in Progress. Stay Tuned!")
    
    replicate_image_tool = ReplicateImageTool(api_token=os.getenv("REPLICATE_API_TOKEN"))

    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        convert_system_message_to_human=True,
        handle_parsing_errors=True,
        temperature=0.4,
        max_tokens=200,
        safety_settings={
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE, 
        },
    )

    word_problem_template = """You are an AI assistant named "AELM-Aristotle" by Mundus tasked with solving the user's logic-based questions. Your goal is to have a natural conversation with a human and be as helpful as possible. If you do not know the answer to a question, You will say "I don't have enough information to respond to that question."
    By default, "Reasoning Tool" is used.
    You are to have an engaging, productive dialogue with your human user. Logically arrive at the solution, and be factual. In your answers, clearly detail the steps involved and give
    the final answer. Provide the response in bullet points. Question: {question} Answer:"""

    math_assistant_prompt = PromptTemplate(
        input_variables=["question"],
        template=word_problem_template
    )

    word_problem_chain = LLMChain(llm=llm, prompt=math_assistant_prompt)
    word_problem_tool = Tool.from_function(
        name="Reasoning Tool",
        func=word_problem_chain.run,
        description="Useful for when you need to answer user with logic-based/reasoning questions."
    )

    problem_chain = LLMMathChain.from_llm(llm=llm)
    math_tool = Tool.from_function(
        name="Calculator",
        func=problem_chain.run,
        description="Useful for when you need to answer numeric questions. This tool is only for math questions and nothing else. Only input math expressions, without text."
    )

    wikipedia = WikipediaAPIWrapper()
    wikipedia_tool = Tool.from_function(
        name="Wikipedia",
        func=wikipedia.run,
        description="A useful tool for searching the Internet to find information on world events, issues, dates, years, etc. Worth using for general topics. Use precise questions."
    )

    ddg_search = DuckDuckGoSearchRun()
    ddg_search_tool = Tool.from_function(
        name="DuckDuckGo Search",
        func=ddg_search.run,
        description="Useful to browse information from the internet to know recent results and information you don't know. Then, tell user the result."
    )

    python_repl = PythonREPL()
    repl_tool = Tool.from_function(
        name="python_repl",
        description="A Python shell. Use this to execute python commands. Input should be a valid python command. You can import and use 'sympy' in python code, if you want to calculate math question. Must not use 'TOOL_CALL' in action input. If you want to see the output of a value, you should print it out with `print(...)`.",
        func=python_repl.run,
    )

    image_generation_tool = Tool.from_function(
        name="Image Generator",
        func=replicate_image_tool.generate_image,
        description="Generates an image based on the provided prompt using Replicate AI. You must return the image url."
    )

    agent = initialize_agent(
        tools=[word_problem_tool, math_tool, wikipedia_tool, ddg_search_tool, repl_tool, image_generation_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        handle_parsing_errors=True
    )
    cl.user_session.set("agent", agent)

@cl.on_message
async def process_user_query(message: cl.Message):
    agent = cl.user_session.get("agent")
    response = await agent.acall(message.content, callbacks=[cl.AsyncLangchainCallbackHandler()])
    await cl.Message(content=response["output"]).send()

# Run the application with `chainlit run main.py -w --port 8000`