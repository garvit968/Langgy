import openai 
import os
import datetime
import requests
import wikipedia
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field
from typing import List
from langchain.tools import tool

from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents import AgentExecutor
from langchain.memory.buffer import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.utils.function_calling import convert_to_openai_function

import panel as pn
import param
from langchain_openai import ChatOpenAI

load_dotenv(find_dotenv())

openai.api_key = os.getenv("OPENAI_API_KEY")

class OpenMeteoInput(BaseModel):
    latitude: float = Field(...,description="Latitude of Place")
    longitude: float = Field(...,description="Longitude of Place")

@tool(args_schema=OpenMeteoInput)
def get_current_temperature(latitude: float, longitude: float)->str:
    ''' Fetch Current Temperature using OpenMeteo API'''
    BASE_URL = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude" : latitude,
        "longitude" : longitude,
        "hourly" : "temperature_2m",
        "forecast_days" : 1
    }

    response = requests.get(BASE_URL, params)
    data = response.json()
    if response.status_code != 200:
        return "Weather API failed."
    data = response.json()
    now = datetime.datetime.utcnow()
    times = [datetime.datetime.fromisoformat(t.replace("Z", "+00:00")) for t in data['hourly']['time']]
    temps = data['hourly']['temperature_2m']
    index = min(range(len(times)), key=lambda i: abs(times[i] - now))
    return f"The current temperature is {temps[index]}°C"

@tool
def search_wiki(query:str)-> str:
    '''Search Wikipedia for query and return summary'''
    titles = wikipedia.search(query)
    summaries = []
    for title in titles[:3]:
        try:
            page = wikipedia.page(title, auto_suggest=True)
            summaries.append(f'**{page.title} **\n{page.summary}')
        except:
           pass
    return '\n\n'.join(summaries) if summaries else "No result found" 

@tool
def create_your_own(query: str) -> str:
    """Reverse the input text as a custom example tool."""
    return f"You sent: {query}. This reverses it: {query[::-1]}"

tools = [get_current_temperature, search_wiki, create_your_own]

pn.extension()

class ConversationalBot(param.Parameterized):  # fixed 'Parameterized' capitalization
    def __init__(self, tools, **params):
        super().__init__(**params)
        self.panels = []
        self.tools = tools
        # Convert tools to OpenAI function definitions correctly
        self.tool_func = [
            convert_to_openai_function(tool.args_schema) if hasattr(tool, "args_schema") else convert_to_openai_function(tool)
            for tool in tools
        ]

        self.llm = ChatOpenAI(temperature=0, model="gpt-4o-mini").bind(functions=self.tool_func)
        self.memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are helpful but sassy assistant."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        self.chain = RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_to_openai_functions(x["intermediate_steps"])
        ) | self.prompt | self.llm | OpenAIFunctionsAgentOutputParser()

        self.executor = AgentExecutor(agent=self.chain, tools=tools, verbose=False, memory=self.memory)

    def interact(self, query):
        result = self.executor.invoke({"input": query})
        self.answer = result['output']
        self.panels.extend([
            pn.Row('User:', pn.pane.Markdown(query, width=500)),
            pn.Row('ChatBot:', pn.pane.Markdown(self.answer, width=500, styles={"background-color": "#f0f0f0"}))
        ])
        return pn.WidgetBox(*self.panels, scroll=True)
    
    
# ========== Launch the Panel Chat App ==========
cb = ConversationalBot(tools)
inp = pn.widgets.TextInput(placeholder='Ask me anything...')
conversation = pn.bind(cb.interact, inp)

tab = pn.Column(
    pn.Row(inp),
    pn.layout.Divider(),
    pn.panel(conversation, loading_indicator=True, height=400),
    pn.layout.Divider()
)

dashboard = pn.Column(
    pn.Row(pn.pane.Markdown('# 🧠 Conversational Agent Bot')),
    pn.Tabs(('Chat', tab))
)

dashboard.servable()
