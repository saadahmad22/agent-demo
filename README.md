# Building a Customer Support Bot with LangGraph 🤖

A comprehensive tutorial for creating AI assistants that can handle flight bookings, hotel reservations, car rentals, and customer service inquiries using LangGraph.

## <a id="table-of-contents"></a>📋 Table of Contents

- [Setup Instructions](#setup-instructions)
- [Project Overview](#project-overview)
- [Code Files Overview](#code-files-overview)
- [Architecture](#architecture)
- [Usage](#usage)
- [Learning Objectives](#learning-objectives)

---

## <a id="setup-instructions"></a>🚀 Setup Instructions

### 1. Download Project Files

**Option A: Download from GitHub**
1. Go to the GitHub repository
2. Click the green "Code" button
3. Select "Download ZIP"
4. Extract the ZIP file to a folder called "Agentic" on your laptop

**Option B: Download Individual Files**
1. Create a folder called "Agentic" on your laptop
2. Download these files into the folder:
    - `demo.ipynb` - Main tutorial notebook
    - `llm_integration.py` - LLM integration module

### 2. Install Python

If you don't have Python installed:
1. Visit [python.org](https://www.python.org/downloads/)
2. Download Python 3.8 or later
3. Run the installer and ensure "Add Python to PATH" is checked
4. Verify installation by running `python --version` in terminal/command prompt

### 3. Install VS Code

1. Download VS Code from [code.visualstudio.com](https://code.visualstudio.com/Download)
2. Install VS Code following the setup wizard
3. Launch VS Code after installation

### 4. Install Required VS Code Extensions

Install these essential extensions:
1. **Markdown Preview Mermaid Support** - For viewing architecture diagrams
2. **Python** - For Python development support
3. **Jupyter** - For running Jupyter notebooks

**Installation steps:**
1. Open VS Code
2. Click the Extensions icon (square icon in left sidebar) or press `Ctrl+Shift+X`
3. Search for each extension name and click "Install"
4. **Restart VS Code** after installing all extensions

### 5. Set Up Replicate API Key

This project uses Replicate for AI language model access:

1. **Create Replicate Account**
    - Visit [replicate.com](https://replicate.com)
    - Click "Sign Up" and create an account

2. **Add Billing Information**
    - Go to your account settings
    - Add a payment method (required for API access)
    - Note: Usage is typically very affordable for learning purposes

3. **Generate API Key**
    - Navigate to [replicate.com/account/api-tokens](https://replicate.com/account/api-tokens)
    - Click "Create Token"
    - Copy your API token (starts with `r8_...`)
    - **Keep this secure** - treat it like a password

### 6. Run the Project

1. Open VS Code
2. Open the "Agentic" folder: `File > Open Folder`
3. Click on `demo.ipynb` to open the notebook
4. When prompted to select a kernel, choose your Python installation
5. Run the notebook cells step by step
6. When prompted for API key, paste your Replicate token

---

## <a id="project-overview"></a>🎯 Project Overview

This project demonstrates building progressively sophisticated AI customer support assistants using LangGraph. You'll learn to create agents that evolve from simple tool-calling bots to production-ready systems with safety mechanisms and specialized workflows.

### What You'll Build

1. **Simple Agent** - Basic tool-calling assistant
2. **Safe Agent** - Added user confirmations for all actions  
3. **Smart Agent** - Conditional interrupts for sensitive operations only
4. **Specialized Agent** - Domain-specific workflows for complex processes

### Key Technologies

- **LangGraph** - Graph-based AI workflow orchestration
- **LangChain** - AI application development framework
- **Replicate** - Cloud-based language model API
- **SQLite** - Local database for travel booking data
- **Python** - Core programming language

---

## <a id="code-files-overview"></a>📁 Code Files Overview

### `demo.ipynb` - Main Tutorial Notebook

The primary learning resource containing four progressive tutorials:

**Part 1: Simple Zero-shot Agent**
- Basic assistant with direct tool execution
- No safety checks or confirmations
- Demonstrates core LangGraph concepts

**Part 2: Adding User Confirmations** 
- Enhanced safety with user approval required for ALL actions
- Interrupt-based workflow control
- Better error handling and context management

**Part 3: Smart Conditional Interrupts**
- Intelligent decision-making about when to interrupt
- Safe operations (search) proceed automatically
- Sensitive operations (booking) require approval

**Part 4: Specialized Workflows** *(Educational framework provided)*
- Architecture for domain-specific assistant workflows
- Production deployment patterns
- Advanced error handling and recovery

### `llm_integration.py` - LLM Integration Module

A comprehensive wrapper for interacting with language models through Replicate:

#### Key Classes and Functions:

**`ReplitLLM` Class**
- Main interface for language model interactions
- Handles authentication and model configuration
- Manages conversation history and context

**Core Methods:**

1. **`__init__(api_key, model)`**
    - Initializes LLM with API credentials
    - Sets up default model configuration
    - Configures environment variables

2. **`prompt_for_api_key()`**
    - Interactive API key setup
    - Validates credentials
    - Provides user-friendly setup experience

3. **`generate_response(prompt, system_prompt, conversation_history, temperature, max_tokens, tools_available)`**
    - Main response generation method
    - Handles tool integration and calling
    - Manages conversation context and history
    - Returns structured response with content and tool calls

4. **`_init_llm()`**
    - Internal method to initialize Replicate LLM instance
    - Handles connection errors gracefully
    - Sets model parameters

5. **`_parse_response_with_tools(raw_response, tools_available)`**
    - Parses LLM responses to extract tool calls
    - Separates natural language from function calls
    - Handles multiple tool call formats
    - Generates natural language descriptions for tool usage

6. **`_parse_tool_args(args_str)`**
    - Extracts and validates tool arguments
    - Handles type conversion (strings, numbers, booleans)
    - Supports both quoted and unquoted parameters

7. **`_generate_tool_description(tool_name, args)`**
    - Creates human-readable descriptions of tool actions
    - Provides context-aware explanations
    - Enhances user experience with clear communication

### Supporting Components

**Database Tools (`demo.ipynb` cells)**
- `setup_database()` - Downloads and configures travel booking database
- `search_flights()` - Searches available flights
- `search_hotels()` - Finds hotel accommodations
- `book_hotel()` - Makes hotel reservations
- `book_car_rental()` - Books car rentals
- `cancel_booking()` - Cancels existing reservations
- `lookup_policy()` - Retrieves company policies
- `fetch_user_flight_information()` - Gets user's current bookings

**Web Search Integration**
- `FreeWebSearch` class - Integrates DuckDuckGo search
- `web_search_tool()` - Provides web search capabilities

**Assistant Classes**
- `SimpleAssistant` - Basic tool-calling agent
- `SafeAssistant` - Enhanced agent with user context and safety

---

## <a id="architecture"></a>🏗️ Architecture

### Graph-Based Workflow

The project uses LangGraph to create stateful, graph-based AI workflows:

```
User Input → Context Loading → AI Processing → Tool Execution → Response
      ↑                                              ↓
      └── User Approval ← Safety Check ← Tool Analysis
```

### State Management

Each conversation maintains:
- **Message History** - Full conversation context
- **User Information** - Current bookings and preferences  
- **Tool Results** - Outputs from database and web searches
- **Approval Status** - User confirmations for actions

### Safety Mechanisms

1. **Universal Interrupts** (Part 2) - Pause before any tool execution
2. **Conditional Interrupts** (Part 3) - Smart decisions about when to pause
3. **User Approval Flow** - Clear confirmation requests
4. **Error Recovery** - Graceful handling of failures

---

## <a id="usage"></a>🎮 Usage

### Running the Tutorial

1. Open `demo.ipynb` in VS Code
2. Execute cells sequentially from top to bottom
3. Follow the interactive prompts
4. Experiment with different queries and scenarios

### Example Interactions

**Safe Searches (No Interruption Needed):**
- "What flights are available from Paris to London?"
- "Show me hotels in Basel"
- "What are your cancellation policies?"

**Sensitive Operations (Requires Approval):**
- "Book hotel ID 123"
- "Cancel my flight reservation"
- "Update my booking to a different date"

### Testing Different Scenarios

The notebook includes comprehensive test functions that demonstrate:
- Basic tool calling
- Safety interrupts
- User approval workflows
- Error handling
- Multi-step processes

---

## <a id="learning-objectives"></a>📚 Learning Objectives

By completing this tutorial, you'll understand:

### Technical Skills
- **Graph-based AI Architecture** - Design complex AI workflows
- **State Management** - Handle conversation context and user data
- **Tool Integration** - Connect AI with databases and APIs
- **Safety Engineering** - Build responsible AI with human oversight
- **Production Patterns** - Scale AI assistants for real-world use

### AI Engineering Concepts
- **LangGraph Framework** - Advanced workflow orchestration
- **Interrupt Patterns** - User control and approval mechanisms
- **Context Awareness** - Maintaining conversation and user state
- **Error Handling** - Robust AI system design
- **Tool Calling** - AI function execution and integration

### Business Applications
- **Customer Service Automation** - 24/7 support capabilities
- **Booking and Reservation Systems** - Travel and hospitality
- **Policy and Information Retrieval** - Knowledge base integration
- **Multi-step Process Automation** - Complex workflow management

---

## 🔧 Troubleshooting

### Common Issues

**API Key Problems:**
- Ensure your Replicate API key is valid and has billing enabled
- Check that you've copied the full token including the `r8_` prefix

**Package Installation:**
- Use `pip install --user` if you encounter permission errors
- Consider using a virtual environment for isolated dependencies

**VS Code Extension Issues:**
- Restart VS Code after installing extensions
- Ensure Python extension can find your Python installation

**Database Issues:**
- Check internet connection for database download
- Verify the `travel2.sqlite` file is in your project folder

### Getting Help

- Review error messages carefully - they often contain helpful information
- Check the console output for detailed error traces
- Ensure all prerequisite software is properly installed

---

## 🎯 Next Steps

After completing this tutorial:

1. **Extend the Tools** - Add weather, news, or restaurant booking capabilities
2. **Enhance Safety** - Implement more sophisticated approval mechanisms  
3. **Add Personalization** - Store and use user preferences
4. **Connect Real APIs** - Integrate with actual booking and payment systems
5. **Deploy to Production** - Scale your assistant for real users

---

**Happy learning! 🚀 Build amazing AI assistants that are both powerful and safe.**