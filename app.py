import streamlit as st
from langchain_groq  import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from dotenv import load_dotenv
from langchain.callbacks import StreamlitCallbackHandler

##Setting up the Streamlit app
st.set_page_config(page_title="Text to Math Problem Solver and Data Search Assistant")
st.title("Text to Math Problem Solver Using Google Gemma2 ")

groq_api_key=st.sidebar.text_input(label="Groq API KEY", type="password")


if not groq_api_key:
    st.info("Please Add Your GROQ API KEY to Continue")
    st.stop()
llm=ChatGroq(model="qwen/qwen3-32b", groq_api_key=groq_api_key)

##Initializing the tools
wikipedia_wrapper=WikipediaAPIWrapper()
wikipedia_tool= Tool(

    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="A Tool for Searching the Internet to find various information on the tool"
)

##Initializing the Math Tool
math_chain= LLMMathChain.from_llm(llm=llm)
calculator=Tool(
    name="calculator",
    func=math_chain.run,
    description="A Tool for answering math related questions. Only Input Mathematical Expressions"
)

prompt="""
You are a agent tasked for solving user's mathematical questions.Logically arrive at the solution and provide detailed explanation and display it pointwise for the question below.
Question:{question}
Answer:
"""

prompt_template=PromptTemplate(
    input_variables=["question"],
    template=prompt
)

##Combine all the tools into chain
chain=LLMChain(llm=llm, prompt=prompt_template)
reasoning_tool=Tool(
    name="Reasoning Tool",
    func=chain.run,
    description="A Tool for answering logic-based and reasoning questions"
)


##initialize the agents
assistant_agent=initialize_agent(

    tools=[wikipedia_tool,calculator, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_error=True
)

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {
            "role": "assistant", 
            "content":"Hi, I'm a Math ChatBot who can answer all your math question"
        }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])



##Starting the Interaction

question= st.text_area("Enter your Question:", "I have 8 oranges and 5 apples. I eat 3 oranges and give away 2 apples. Then I buy 4 more apples and 6 more oranges. How many total fruits do I have now?")
if st.button("Find My Answer"):
    if question:
        with st.spinner("Generate response"):
            st.session_state.messages.append({"role":"user", "content":question})
            st.chat_message("user").write(question)

            st_cb= StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response=assistant_agent.run(st.session_state.messages, callbacks=[st_cb])
            st.session_state.messages.append({'role':'assistant', "content":response})
            st.write('### Response:')
            st.success(response)

    else:
        st.warning("Please Enter the Question:")