import os
import sys
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
import google.generativeai as genai

# Resolve paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# Load environment variables
load_dotenv(os.path.join(BASE_DIR, ".env"))
# Fallback to system environment if needed
api_key = os.environ.get("GEMINI_API_KEY")

if api_key:
    genai.configure(api_key=api_key)
else:
    # We will log a warning, but we won't crash immediately on import so indexer can run
    print("WARNING: GEMINI_API_KEY not found in environment or .env file. The agent will require this key to run.")

from assistant.retriever import search
from assistant.tools import check_stock, find_parts_by_vehicle, create_order
from assistant.prompts import SYSTEM_PROMPT

def dispatch_tool(name, args):
    """Executes the tool with the provided arguments and returns JSON-serializable output."""
    if name == "check_stock":
        return check_stock(**args)
    elif name == "find_parts_by_vehicle":
        return find_parts_by_vehicle(**args)
    elif name == "create_order":
        return create_order(**args)
    else:
        return {"error": f"Unknown tool: {name}"}

def process_response_with_tools(chat, response):
    """Handles the function calling loop recursively if the model issues tool calls."""
    while True:
        # Check if the response contains any function calls
        function_calls = []
        # If response has parts, check them
        if response.parts:
            for part in response.parts:
                if fn := getattr(part, "function_call", None):
                    function_calls.append(fn)

        if not function_calls:
            # No tool calls, we can return the text response
            break

        # Process each tool call
        response_parts = []
        for fn in function_calls:
            tool_name = fn.name
            tool_args = dict(fn.args)
            # Execute the tool
            result = dispatch_tool(tool_name, tool_args)
            
            # Create a function response part
            response_part = genai.protos.Part(
                function_response=genai.protos.FunctionResponse(
                    name=tool_name,
                    response={"result": result}
                )
            )
            response_parts.append(response_part)

        # Send the tool output back to the model
        response = chat.send_message(response_parts)

    return response

def query_agent(chat, user_input: str) -> str:
    """Helper function to run a query against the agent with context injection and tool handling."""
    # 1. Retrieve context items from ChromaDB
    try:
        context_items = search(user_input, top_k=5)
        context_str = "\n".join([
            f"- {i['sku']}: {i['name']} | Price: ₹{i['price_inr']} | Stock: {i['stock']} | Fits: {i['vehicle_fitment']}"
            for i in context_items
        ])
    except Exception as e:
        context_str = f"Error retrieving context: {e}"

    # 2. Formulate augmented input
    augmented_input = (
        f"[Relevant catalogue items]\n{context_str}\n\n"
        f"[Dealer query]\n{user_input}"
    )

    # 3. Send message to chat and handle any tool execution
    response = chat.send_message(augmented_input)
    response = process_response_with_tools(chat, response)
    
    return response.text

def get_agent_model():
    """Initializes and returns the Gemini GenerativeModel with tools and system instruction."""
    # List of functions to be exposed as tools
    tools = [check_stock, find_parts_by_vehicle, create_order]
    
    return genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        tools=tools,
        system_instruction=SYSTEM_PROMPT
    )

def run_agent_cli():
    """Starts the interactive command-line interface for the dealer assistant."""
    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable is not set. Please set it in your environment or .env file.")
        sys.exit(1)

    model = get_agent_model()
    chat = model.start_chat()
    
    print("="*60)
    print("VIKMO Dealer Assistant CLI")
    print("Type 'exit' or 'quit' to end the session.")
    print("="*60)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
                
            response_text = query_agent(chat, user_input)
            print(f"\nAssistant: {response_text}")
        except KeyboardInterrupt:
            print("\nSession ended.")
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    run_agent_cli()
