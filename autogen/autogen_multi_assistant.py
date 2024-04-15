import os
import openai
import autogen
from openai import OpenAI
from dotenv import load_dotenv
from autogen import UserProxyAgent, GroupChat, GroupChatManager
from autogen.agentchat.contrib.gpt_assistant_agent import GPTAssistantAgent
from tools.searchtool import Search

load_dotenv()

openai_model: str = os.environ.get("OPENAI_MODEL")
openai.api_key = os.environ.get("OPENAI_API_KEY")
# create client for OpenAI
client = OpenAI(api_key=openai.api_key)
search_m: Search = Search()  # get instance of search to query corpus
search_a: Search = Search()  # get instance of search to query corpus
search_o: Search = Search()  # get instance of search to query corpus

config_list_gpt4 = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": ["gpt-4-turbo"],
    },
)
llm_config = {"config_list": config_list_gpt4, "cache_seed": 42}

# Function to perform a Microsoft
def microsoft_retrieval(query):
    print("calling Microsoft search")
    search_result = search_m.search_hybrid(query, "Microsoft")
    return search_result


# Function to perform a AWS
def aws_retrieval(query):
    print("calling AWS search")
    search_result = search_a.search_hybrid(query, "AWS")
    return search_result


# Function to perform a Oracle
def oracle_retrieval(query):
    print("calling Oracle search")
    search_result = search_o.search_hybrid(query, "Oracle")
    return search_result


if __name__ == "__main__":
    # Retrieve an existing assistant already setup as an OpenAI Assistant
    # this is OpenAI Assistant stuff
    microsoft_retriever = client.beta.assistants.retrieve(
        assistant_id="asst_dANZJ31IHkMbvEHU0HDH3Zxt",
    )

    aws_retriever = client.beta.assistants.retrieve(
        assistant_id="asst_ekigD4AX2R3ij6tqTIYvtiVa",
    )

    oracle_retriever = client.beta.assistants.retrieve(
        assistant_id="asst_YBi5VOql08zBLzR3g7A4NnrN",
    )

    planner_assistant = client.beta.assistants.retrieve(
        assistant_id="asst_HOeqxoPgbXdIzhepX8tvmekl",
    )

    # define the config including the tools that the assistant has access to
    # this will be used by the GPTAssistant Agent that is Shadow Retriever
    microsoft_retriever_config = {
        "config_list": llm_config,
        "assistant_id": microsoft_retriever.id,
        "tools": [
            {
                "type": "function",
                "function": microsoft_retrieval,
            }
        ],
    }

    aws_retriever_config = {
        "config_list": llm_config,
        "assistant_id": aws_retriever.id,
        "tools": [
            {
                "type": "function",
                "function": aws_retrieval,
            }
        ],
    }

    oracle_retriever_config = {
        "config_list": llm_config,
        "assistant_id": oracle_retriever.id,
        "tools": [
            {
                "type": "function",
                "function": oracle_retrieval,
            }
        ],
    }

    planner_assistant_config = {
        "config_list": llm_config,
        "assistant_id": planner_assistant.id,
    }

    user_proxy = UserProxyAgent(
        name="Admin",
        human_input_mode="ALWAYS",
        description="User Proxy Admin that interacts with the Planner",
        system_message="A human admin. Interact with the planner to discuss the plan. Plan execution needs to be approved by this admin.",
        code_execution_config=False,
        max_consecutive_auto_reply=10,
    )

    # this is autogen stuff defining the agent that is going to be in the group
    microsoft_retriever_agent = GPTAssistantAgent(
        name="MicrosoftRetriever",
        description="Retrieves relevant information on Microsoft",
        instructions=None,
        llm_config={
            "config_list": config_list_gpt4,
        },
        assistant_config=microsoft_retriever_config
    )

    # this is autogen stuff defining the agent that is going to be in the group
    aws_retriever_agent = GPTAssistantAgent(
        name="AWSRetriever",
        description="Retrieves relevant information on Amazon",
        instructions=None,
        llm_config={
            "config_list": config_list_gpt4,
        },
        assistant_config=aws_retriever_config
    )

    # this is autogen stuff defining the agent that is going to be in the group
    oracle_retriever_agent = GPTAssistantAgent(
        name="OracleRetriever",
        description="Retrieves relevant information on Oracle",
        instructions=None,
        llm_config={
            "config_list": config_list_gpt4,
        },
        assistant_config=oracle_retriever_config
    )

    # this is autogen stuff defining the agent that is going to be in the group
    planner_agent = GPTAssistantAgent(
        name="Planner",
        description="Planner agent",
        instructions=None,
        llm_config={
            "config_list": config_list_gpt4,
        },
        assistant_config=planner_assistant_config
    )

    microsoft_retriever_agent.register_function(
        function_map={
            "microsoft_retrieval": microsoft_retrieval,
        }
    )

    aws_retriever_agent.register_function(
        function_map={
            "aws_retrieval": aws_retrieval,
        }
    )

    oracle_retriever_agent.register_function(
        function_map={
            "oracle_retrieval": oracle_retrieval,
        }
    )

    # groupchat = GroupChat(agents=[user_proxy, planner_agent, microsoft_retriever_agent, aws_retriever_agent, oracle_retriever_agent,  linkedin_writer_agent], messages=[], max_round=30)
    groupchat = GroupChat(
        agents=[
            user_proxy,
            planner_agent,
            microsoft_retriever_agent,
            aws_retriever_agent,
            oracle_retriever_agent,
        ],
        messages=[],
        max_round=30,
    )
    manager = GroupChatManager(
        groupchat=groupchat, llm_config=llm_config
    )

    print("initiating chat")

    user_proxy.initiate_chat(
        manager,
        message="""
            Find information on Oracle and list what you found.
            """
    )
