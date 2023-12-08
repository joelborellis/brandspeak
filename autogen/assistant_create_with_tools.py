import os
import openai
import time
from halo import Halo
from openai import OpenAI
from dotenv import load_dotenv
from backend.tools.searchtool import Search
import json

load_dotenv()

openai_model: str = os.environ.get("OPENAI_MODEL")
openai.api_key = os.environ.get("OPENAI_API_KEY")
# create client for OpenAI
client = OpenAI(api_key=openai.api_key)
search: Search = Search()  # get instance of search to query corpus

###     file operations

def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as infile:
        return infile.read()
    
###     API functions

def upload_file(filename):
        # Upload a file with an "assistants" purpose
        file = client.files.create(
            file=open(filename, "rb"),
            purpose='assistants'
            )
        return file

# Function to perform a Shadow Search
def shadow_search(query):
    search_result = search.search_hybrid(query)
    return search_result

# Function to handle tool output submission
def submit_tool_outputs(thread_id, run_id, tools_to_call):
    tool_output_array = []
    for tool in tools_to_call:
        output = None
        tool_call_id = tool.id
        function_name = tool.function.name
        function_args = tool.function.arguments

        if function_name == "shadow_search":
            print(json.loads(function_args)["query"])
            output = search.search_hybrid(query=json.loads(function_args)["query"])

        if output:
            tool_output_array.append({"tool_call_id": tool_call_id, "output": output})

    return client.beta.threads.runs.submit_tool_outputs(
        thread_id=thread_id,
        run_id=run_id,
        tool_outputs=tool_output_array
    )


def create_assistant(file_id, assistant_prompt_instruction):
     while True:
            try:
                assistant = client.beta.assistants.create(
                        name="Shadow Tool Assistant",
                        instructions=assistant_prompt_instruction,
                        model="gpt-4-1106-preview",
                        tools=[{
                                    "type": "function",
                                    "function": {
                                        "name": "shadow_search",
                                        "description": "Get information from shadows corpus of sales methodology books.",
                                        "parameters": {
                                            "type": "object",
                                            "properties": {
                                                "query": {"type": "string", "description": "The search query to use. For example: 'I have a first meeting with a prospect - what do I need to find out and what are the most important things I need to relate to them?'"},
                                            },
                                            "required": ["query"]
                                        }
                                    }
                                },
                                {"type": "retrieval"}],
                        file_ids=[file_id],
                    )             

                return assistant
            except Exception as yikes:
                print(f'\n\nError communicating with OpenAI: "{yikes}"')
                exit(5)

# Function to wait for a run to complete
def wait_for_run_completion(thread_id, run_id):
    while True:
        time.sleep(1)
        run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
        print(f"Current run status: {run.status}")
        if run.status in ['completed', 'failed', 'requires_action']:
            return run
        
# Function to print messages from a thread
def print_messages_from_thread(thread_id):
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    for msg in reversed(messages.data):
        print(f"{thread_id}:  {msg.role}: {msg.content[0].text.value}")

if __name__ == '__main__':
    # Read the system messgae from file
    #system_message = open_file('./backend/prompts/system_insights.md')
    assistant_prompt_instruction = """You are a sales expert. 
            Your goal is to provide answers based on information from retrieved files or the corpus of sales data from Shadow Seller. 
            You must use the provided shadow search API function to find relevant information in the corpus. 
            You should never use your own knowledge to answer questions.
            """

    # upload a file
    file = upload_file("./data/Joel_Borellis_RESUME.pdf")

    # get an assistant
    assistant = create_assistant(file.id, assistant_prompt_instruction)

    thread = client.beta.threads.create()

    while True:
     # Get user query
        query = input('\n\nQUERY: ').strip()
        if query.lower() == 'exit':
            exit(0)
        
        

        message = client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=query
            )
        run = client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant.id,
                )
        
        run = wait_for_run_completion(thread.id, run.id)
        
        if run.status == 'failed':
            print(run.error)
            continue
        elif run.status == 'requires_action':
            run = submit_tool_outputs(thread.id, run.id, run.required_action.submit_tool_outputs.tool_calls)
            run = wait_for_run_completion(thread.id, run.id)

        # Print messages from the thread
        print_messages_from_thread(thread.id)

        