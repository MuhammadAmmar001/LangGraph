from dotenv import load_dotenv
from typing import Annotated,Literal
from typing_extension import TypedDict
from langgraph.graph import StateGraph,START,END
from langgraph.graph.messages import add_messages
from langchain_openai import ChatOpenAI
from pydantic import BaseModel,Field

load_dotenv(override=True)

llm = ChatOpenAI(
    model = "meta-llama/llama-4-scout:free" ,
    base_url = "https://openrouter.ai/api/v1"
)

class State(TypedDict):
    messages:Annotated[list,add_messages]
    message_class : str | None

class MessageClassifier(BaseModel):
    message_class:Literal["Emotional","Logical"] = Field(...,description="Assess the message and classify it as Logical or Emotional")

def classifier(state:State):
    last_message = state["messages"][-1]
    message_classifier_llm = llm.with_structured_output(MessageClassifier)
    response = message_classifier_llm.invoke([
        {'role':'system','content':'You are a message classifying agent. You task is to classify the given message into either Logical or Emotional.'},
        {'role':'user','content':last_message.content}
    ])
    return {"message_class":response.message_class}

def router(state:State):
    message_type = state["message_class"] or "Logical"
    if message_type.lower() == "emotional":
        return {"next":"Emotional"}
    return {"next":"Logical"}

def emotional_agent(state:State):
    last_message = state["messages"][-1]
    SYSTEM_PROMPT = """ You are an emotional consoling agent. You understand te emotional state and feeling of the given message. Based on the type of emotion, you respond in the best way to cater and console the situation. Keep your response concise and filled with harmony.
    """
    USER_PROMPT = last_message.content

    emotional_response = llm.invoke([{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":USER_PROMPT}])
    
    return {"messages":[{"role":"assistant","content":emotional_response.content}]}

def logical_agent(state:State):
    last_message = state["messages"][-1]
    SYSTEM_PROMPT = """ You are a professional assistant that specializes in solving problems of all scales with logic and facts. You break down the problem and solve the problem step by step with explanation. Your tone is professional and sounding confident so as to convey an impactful message.
    """
    USER_PROMPT = last_message.content

    logical_response = llm.invoke([{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":USER_PROMPT}])
    
    return {"messages":[{"role":"assistant","content":logical_response.content}]}


graph_builder = StateGraph()

graph_builder.add_node("classifier",classifier)
graph_builder.add_node("router",router)
graph_builder.add_node("emotional_agent",emotional_agent)
graph_builder.add_node("logical_agent",logical_agent)

graph_builder.add_edge(START,"classifier")
graph_builder.add_edge("classifier","router")
graph_builder.add_conditional_edge(
    "router",
    lambda state: state.get("message_class"),
    {"Emotional":"emotional_agent","Logical":"logical_agent"}
    )
graph_builder.add_edge("emotional",END)
graph_builder.add_edge("logical",END)


graph = graph_builder.compile()

def run_workflow():
    initial_state = {"messages":[],"message_class":None}

    while True:

        user_input = input("Enter your problem here: ")
        if user_input == "exit":
            print("Exiting System")
            break

        initial_state["messages"] = initial_state.get("messages",[]) + [{"role":"user","content":user_input}]

        initial_state = graph.invoke(initial_state)

        if initial_state.get("messages") and len(initial_state.get("messages")) > 0:
            print(f"Assistant : {initial_state["messages"][-1].content}")

if __name__ == "__main__":
    run_workflow()