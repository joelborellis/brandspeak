import os
import openai
import time
from openai import OpenAI
from backend.tools.searchtool import Search
import json
from dotenv import load_dotenv

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
    
def create_json_object(id, message_content, role, created_at, finish_reason="stop", index=0, model="gpt-4-1106-preview"):
    json_object = {
        "id": id,
        "choices": [
            {
                "finish_reason": finish_reason,
                "index": index,
                "message": {
                    "content": message_content,
                    "role": role,
                    "function_call": None,
                    "tool_calls": None
                }
            }
        ],
        "created": created_at,
        "model": model,
        "object": "chat.completion",
        "system_fingerprint": "",
        "usage": {
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "total_tokens": 0,
        }
    }

    return json.dumps(json_object, indent=2)
    
###     API functions

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

def get_assistant():
     while True:
            try:
                # Retrieve an existing assistant
                assistant = client.beta.assistants.retrieve(
                        assistant_id="asst_CDgesnP9G5fWP15UVBeQQfUX",
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
        print(f"{msg.thread_id}:  {msg.role}: {msg.content[0].text.value}")
        if msg.role == "assistant":
            m_json = create_json_object(msg.thread_id, msg.content[0].text.value, msg.role, msg.created_at)
            print(m_json)

if __name__ == '__main__':
    # Read the system messgae from file
    #system_message = open_file('./backend/prompts/system_insights.md')
    # get an assistant
    assistant = get_assistant()
    print(assistant.name)
    # create a thread
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