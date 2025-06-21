import os
from typing import Optional, Dict, List, Any
from langchain_community.llms import Replicate
import re

# Default model configuration
DEFAULT_MODEL = "meta/meta-llama-3-8b-instruct"

class ReplitLLM:
    """Simple LLM integration using LangChain's Replicate provider."""
    
    def __init__(self, api_key: str = None, model: str = None):
        """Initialize the LLM with API key and model selection"""
        self.api_key = api_key or os.getenv("REPLICATE_API_TOKEN")
        self.model = model or DEFAULT_MODEL
        self.conversation_history = []
        
        # Set environment variable for Replicate
        if self.api_key:
            os.environ["REPLICATE_API_TOKEN"] = self.api_key
        
        # Initialize the LangChain Replicate LLM
        self.llm = None
        if self.api_key:
            self._init_llm()
    
    def _init_llm(self):
        """Initialize the Replicate LLM instance"""
        try:
            self.llm = Replicate(
                model=self.model,
                model_kwargs={"temperature": 0.7, "max_length": 1024, "top_p": 1}
            )
        except Exception as e:
            print(f"❌ Failed to initialize LLM: {e}")
            self.llm = None
    
    def prompt_for_api_key(self):
        """Simplified API key prompt"""
        print("\n🔑 LLM Setup:")
        api_key = input("📝 Enter your Replicate API key: ").strip()
        
        if api_key:
            self.api_key = api_key
            os.environ["REPLICATE_API_TOKEN"] = self.api_key
            self._init_llm()
            print(f"✅ Using model: {self.model}")
            return True
        else:
            print("❌ No API key provided. Please set the REPLICATE_API_TOKEN environment variable or provide it directly.")
            return False
    
    def generate_response(self, 
                        prompt: str, 
                        system_prompt: Optional[str] = None,
                        conversation_history: Optional[List[Dict[str, str]]] = None,
                        temperature: float = 0.7,
                        max_tokens: int = 1024,
                        tools_available: list = None) -> Dict[str, Any]:
        """Generate a response from the LLM through LangChain's Replicate integration"""
        
        if not self.llm:
            raise RuntimeError(
                "LLM not initialized. Please provide a valid API key or run prompt_for_api_key()."
            )
        
        # Update system prompt with tools if provided
        enhanced_system_prompt = (
            system_prompt or 
            "You are a helpful and friendly AI customer support assistant. "
            "Always respond with natural, conversational language. "
            "Explain what you're doing in a helpful way."
        )
        
        if tools_available:
            enhanced_system_prompt += (
                "\n\nYou have access to these tools:\n"
            )
            for tool in tools_available:
                enhanced_system_prompt += f"- {tool.name}: {tool.__doc__ or 'No description'}\n"
            
            enhanced_system_prompt += (
                "\n\nIMPORTANT: When you need to use a tool, ALWAYS:\n"
                "1. First provide a helpful natural language response explaining what you're doing\n"
                "2. Then call the tool using this format: TOOL_CALL: tool_name(arg1='value1', arg2='value2')\n"
                "\nExamples:\n"
                "User: 'Search for flights from Paris to London'\n"
                "Assistant: I'll search for flights from Paris to London for you.\n"
                "TOOL_CALL: search_flights(departure_airport='CDG', arrival_airport='LHR')\n"
                "\nUser: 'Book hotel 123'\n"
                "Assistant: I'll book hotel ID 123 for you right away.\n"
                "TOOL_CALL: book_hotel(hotel_id=123)\n"
                "\nNEVER respond with just a bare function call. Always include helpful natural language."
            )
        
        # Build the full prompt with system prompt and conversation history
        full_prompt = ""
        
        # Add system prompt
        if enhanced_system_prompt:
            full_prompt += f"System: {enhanced_system_prompt}\n\n"
        
        # Add conversation history - extract from LangChain message format
        if conversation_history and isinstance(conversation_history, str):
            full_prompt += conversation_history
        
        # Add current prompt
        full_prompt += f"User: {prompt}\nAssistant: "
        
        try:
            # Update model kwargs with current parameters
            self.llm.model_kwargs.update({
                "temperature": temperature,
                "max_length": max_tokens
            })
            
            # Generate response
            raw_response = self.llm.invoke(full_prompt)
            
            # Parse response and extract tool calls
            content, tool_calls = self._parse_response_with_tools(raw_response, tools_available)
            
            # Save to history
            self.conversation_history.append(("user", prompt))
            self.conversation_history.append(("assistant", content))
            
            return {
                "content": content,
                "tool_calls": tool_calls
            }
            
        except Exception as e:
            print(f"❌ API request failed: {str(e)}")
            
            raise RuntimeError(
                "Failed to connect to the LLM API. Please check your API key and network connection. "
            )
    
    def _parse_response_with_tools(self, raw_response: str, tools_available=None) -> tuple[str, List[Dict[str, Any]]]:
        """Parse LLM response to separate natural language content from tool calls"""
        tool_calls = []
        
        if not tools_available:
            return raw_response, tool_calls
            
        # Create a mapping of tool names to tool objects
        tool_map = {tool.name: tool for tool in tools_available}
        
        # Check if the response is ONLY a tool call (no natural language)
        raw_response = raw_response.strip()
        
        # Pattern to match standalone function calls like: tool_name(arg1="value1", arg2="value2")
        standalone_tool_pattern = r'^(\w+)\s*\(\s*([^)]*)\s*\)$'
        standalone_match = re.match(standalone_tool_pattern, raw_response)
        
        if standalone_match and standalone_match.group(1) in tool_map:
            # This is a standalone tool call - provide natural language description
            func_name, args_str = standalone_match.groups()
            
            # Parse arguments
            args = self._parse_tool_args(args_str)
            
            tool_calls.append({
                "name": func_name,
                "args": args,
                "id": f"call_{len(tool_calls)}"
            })
            
            # Generate natural language content based on the tool call
            content = self._generate_tool_description(func_name, args)
            
            return content, tool_calls
        
        # Check for multiple standalone tool calls or mixed format
        lines = raw_response.split('\n')
        processed_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this line is a standalone tool call
            match = re.match(standalone_tool_pattern, line)
            if match and match.group(1) in tool_map:
                func_name, args_str = match.groups()
                args = self._parse_tool_args(args_str)
                
                tool_calls.append({
                    "name": func_name,
                    "args": args,
                    "id": f"call_{len(tool_calls)}"
                })
                
                # Replace the tool call with natural language
                description = self._generate_tool_description(func_name, args)
                processed_lines.append(description)
            else:
                processed_lines.append(line)
        
        if processed_lines:
            content = " ".join(processed_lines)
        else:
            content = raw_response
        
        # Look for TOOL_CALL: prefix pattern
        tool_call_pattern = r'TOOL_CALL:\s*(\w+)\s*\(\s*([^)]*)\s*\)'
        content_parts = []
        current_pos = 0
        
        for match in re.finditer(tool_call_pattern, raw_response):
            # Add content before this tool call
            content_parts.append(raw_response[current_pos:match.start()].strip())
            
            func_name, args_str = match.groups()
            if func_name in tool_map:
                args = self._parse_tool_args(args_str)
                
                tool_calls.append({
                    "name": func_name,
                    "args": args,
                    "id": f"call_{len(tool_calls)}"
                })
            
            current_pos = match.end()
        
        # Add remaining content
        content_parts.append(raw_response[current_pos:].strip())
        
        # Join content parts and clean up
        content = " ".join(part for part in content_parts if part)
        
        # If no natural language content, generate some
        if not content and tool_calls:
            content = self._generate_tool_description(tool_calls[0]["name"], tool_calls[0]["args"])
        
        return content or raw_response, tool_calls
    
    def _parse_tool_args(self, args_str: str) -> Dict[str, Any]:
        """Parse tool arguments from string format"""
        args = {}
        if args_str.strip():
            # Handle both quoted and unquoted arguments
            # Pattern for key=value pairs with optional quotes
            arg_pattern = r'(\w+)\s*=\s*(["\']?)([^"\']*?)\2(?:\s*,\s*|$)'
            arg_matches = re.findall(arg_pattern, args_str)
            
            for key, quote, value in arg_matches:
                # Try to convert to appropriate type
                if value.isdigit():
                    args[key] = int(value)
                elif value.replace('.', '').isdigit():
                    args[key] = float(value)
                elif value.lower() in ['true', 'false']:
                    args[key] = value.lower() == 'true'
                else:
                    args[key] = value
        
        return args
    
    def _generate_tool_description(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Generate natural language description for tool calls"""
        if tool_name == "search_flights":
            departure = args.get("departure_airport", "your departure city")
            arrival = args.get("arrival_airport", "your destination")
            return f"I'll search for flights from {departure} to {arrival} for you."
        
        elif tool_name == "search_hotels":
            location = args.get("location", args.get("city", "your destination"))
            return f"Let me search for hotels in {location}."
        
        elif tool_name == "book_hotel":
            hotel_id = args.get("hotel_id", "the selected hotel")
            return f"I'll book hotel ID {hotel_id} for you."
        
        elif tool_name == "book_car_rental":
            rental_id = args.get("rental_id", "the selected car")
            return f"I'll book car rental ID {rental_id} for you."
        
        elif tool_name == "cancel_booking":
            booking_type = args.get("booking_type", "booking")
            booking_id = args.get("booking_id", "")
            return f"I'll cancel your {booking_type} {booking_id} for you."
        
        elif tool_name == "lookup_policy":
            return "Let me look up our company policies for you."
        
        elif tool_name == "fetch_user_flight_information":
            return "Let me check your current flight bookings."
        
        elif tool_name == "web_search_tool":
            query = args.get("query", "your request")
            return f"I'll search the web for information about {query}."
        
        else:
            return f"I'll use the {tool_name} tool to help you."
