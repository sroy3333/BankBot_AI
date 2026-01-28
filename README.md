# BankBot AI - Intelligent Banking Assistant

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![AI](https://img.shields.io/badge/AI-Powered-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-Natural%20Language%20Processing-4285F4?style=for-the-badge&logo=google&logoColor=white)
![LLM](https://img.shields.io/badge/LLM-Large%20Language%20Model-412991?style=for-the-badge&logo=openai&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

## 📋 Project Description

BankBot AI is an intelligent, AI-powered conversational banking assistant designed to provide secure and efficient banking services through natural language interactions. The system leverages advanced Natural Language Processing (NLP) and Large Language Model (LLM) technologies to understand user intents, extract entities, and perform banking operations seamlessly.

The chatbot offers a comprehensive suite of banking functionalities including balance inquiries, money transfers, card management, and ATM locator services, all while maintaining robust security measures and detailed analytics for administrative oversight.

## ✨ Features

- **💬 Conversational Interface**: Natural language interaction for banking operations
- **🔐 Secure Authentication**: User authentication and session management
- **💰 Balance Inquiry**: Real-time account balance checking
- **💸 Money Transfer**: Secure fund transfers between accounts with validation
- **🚫 Card Management**: Block/unblock credit and debit cards
- **📍 ATM Locator**: Find nearby ATMs based on location
- **🤖 Intent Recognition**: Intelligent detection of user intentions
- **🔍 Entity Extraction**: Automatic extraction of account numbers, amounts, and other entities
- **📊 Admin Analytics**: Comprehensive dashboards for chat and query analytics
- **📈 Real-time Monitoring**: Track user queries, intents, and system performance
- **❓ FAQ Management**: Dynamic FAQ system for common banking questions
- **🛡️ Safety Guards**: Content moderation and security validation
- **📝 Logging & Auditing**: Complete conversation logging for compliance

## 🔬 Techniques Used

### Natural Language Processing (NLP)

- **Intent Classification**: Multi-class classification using transformer-based models
- **Named Entity Recognition (NER)**: Extraction of banking entities (account numbers, amounts, card types)
- **Text Preprocessing**: Tokenization, normalization, and cleaning
- **Semantic Similarity**: FAQ matching using sentence embeddings
- **Dialogue State Management**: Context-aware conversation flow handling

### Prompt Engineering

- **Dynamic Prompt Construction**: Context-aware prompts for LLM interactions
- **Few-shot Learning**: Template-based prompt engineering for specific banking tasks
- **System Prompts**: Structured system instructions for consistent responses
- **Context Management**: Conversation history integration for coherent dialogues

### LLM-based Text Generation

- **Contextual Responses**: Generate human-like responses based on user queries
- **Query Understanding**: Deep comprehension of banking-related questions
- **Multi-turn Conversations**: Maintain context across conversation turns
- **Fallback Mechanisms**: Graceful handling of out-of-scope queries

## 🛠️ Tech Stack

### Programming Language

- **Python 3.11+**: Core development language

### Libraries / Frameworks

#### Frontend & UI
- **Streamlit**: Interactive web interface and admin panel
- **Plotly Express**: Data visualization and analytics charts

#### Data & Database
- **SQLite**: Local database for accounts, transactions, and logs
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing

#### NLP & Machine Learning
- **Transformers (Hugging Face)**: Pre-trained language models
- **Sentence-Transformers**: Semantic text similarity
- **scikit-learn**: ML utilities and preprocessing
- **NLTK/spaCy**: Text processing tools

#### API & Integration
- **Groq API**: LLM inference endpoints
- **Requests**: HTTP client for external services

### AI / ML Technologies

- **Intent Classification Models**: Transformer-based classifiers
- **Entity Recognition**: Custom NER pipelines
- **Embedding Models**: Sentence-BERT for semantic search
- **Safety Filters**: Content moderation models

## 🤖 LLM Details

### Model Architecture

The system utilizes **transformer-based Large Language Models (LLMs)** for natural language understanding and generation:

- **Base Architecture**: Decoder-only transformer models (GPT-style)
- **Context Window**: Supports extended conversation context
- **Fine-tuning**: Domain adaptation for banking terminology

### Configurable LLM

The LLM integration is **fully configurable** and supports multiple providers:

- **Groq API**: Primary LLM service provider
- **Model Selection**: Easily switch between different model versions
- **Custom Endpoints**: Configure custom LLM endpoints
- **Fallback Options**: Multiple model fallback support

To configure the LLM, modify the settings in `services/llm_service.py`:

```python
# Example configuration
LLM_PROVIDER = "groq"  # Options: groq, openai, anthropic, custom
MODEL_NAME = "mixtral-8x7b-32768"  # Configurable model
API_KEY = "your_api_key_here"  # Set via environment variable
```

## 📁 Project Structure

```
BankBot_AI/
│
├── admin_chat_analytics.py      # Chat analytics dashboard
├── admin_panel.py                # Main admin interface
├── admin_query_analytics.py      # Query analytics dashboard
├── main_app.py                   # Main Streamlit application
├── requirements.txt              # Python dependencies
│
├── database/                     # Database modules
│   ├── bank_crud.py             # CRUD operations for banking
│   ├── db.py                    # Database connection and setup
│   ├── logger.py                # Database logging utilities
│   ├── security.py              # Security and authentication
│   └── bankbot.db               # SQLite database file
│
├── dialogue_manager/             # Conversation management
│   ├── dialogue_handler.py      # Main dialogue processing
│   ├── dialogue_state.py        # State management
│   └── stories.py               # Predefined conversation flows
│
├── nlu_engine/                   # NLP processing
│   ├── entity_extractor.py      # Entity extraction logic
│   ├── infer_intent.py          # Intent classification
│   ├── train_intent.py          # Model training scripts
│   ├── intents.json             # Intent definitions
│   └── entities.json            # Entity schemas
│
├── router/                       # Query routing
│   └── query_router.py          # Route queries to handlers
│
├── services/                     # Core services
│   ├── atm_service.py           # ATM location services
│   ├── chat_analytics.py        # Analytics data processing
│   ├── chat_logger.py           # Conversation logging
│   ├── llm_service.py           # LLM integration
│   ├── query_analytics.py       # Query analysis
│   └── safety_guard.py          # Security validation
│
├── knowledge_base/               # Knowledge management
│   └── faqs.csv                 # FAQ database
│
└── logs/                         # Application logs
    └── chat_logs.csv            # Conversation history
```

## 📦 Installation Steps

### Prerequisites

- Python 3.11 or higher
- pip package manager
- Virtual environment (recommended)

### Step-by-Step Installation

1. **Clone the Repository**

```bash
git clone https://github.com/sroy3333/BankBot_AI.git
cd BankBot_AI
```

2. **Create Virtual Environment**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

4. **Set Up Environment Variables**

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
DATABASE_PATH=database/bankbot.db
LOG_LEVEL=INFO
```

5. **Initialize Database**

```bash
python database/db.py
```

6. **Train NLP Models (First-time Setup)**

```bash
python nlu_engine/train_intent.py
```

## 🚀 How to Run the Project Locally

### Running the Main Application

```bash
streamlit run main_app.py
```

The application will start on `http://localhost:8501`

### Running the Admin Panel

```bash
streamlit run admin_panel.py
```

The admin panel will be accessible on `http://localhost:8502`

### Command-Line Options

```bash
# Run with custom port
streamlit run main_app.py --server.port 8080

# Run with custom host
streamlit run main_app.py --server.address 0.0.0.0

# Run in development mode
streamlit run main_app.py --server.runOnSave true
```

## 📚 Usage Examples

### User Chat Interface

1. Open the main application
2. Enter your account number for authentication
3. Start chatting with BankBot using natural language:
   - "What's my account balance?"
   - "Transfer ₹5000 to account 67890"
   - "Block my credit card"
   - "Find ATMs near Connaught Place"

### Admin Panel

1. Open the admin panel
2. Navigate through tabs:
   - **Training Data**: Edit NLP training examples
   - **FAQs Manager**: Add/edit frequently asked questions
   - **User Queries**: View all user interactions
   - **Chat Analytics**: Visualize intent distribution
   - **Query Analytics**: Analyze query patterns

## 🎓 Certification Use Case

### Infosys Certification Context

This project demonstrates proficiency in:

1. **AI/ML Development**: Implementation of NLP pipelines and LLM integration
2. **Software Engineering**: Modular architecture, clean code practices
3. **Database Design**: Efficient schema design for banking operations
4. **Security Implementation**: Authentication, validation, and logging
5. **Data Analytics**: Real-time analytics and visualization
6. **UI/UX Design**: User-friendly interfaces with Streamlit
7. **Project Management**: Complete end-to-end application development

### Key Learning Outcomes

- Understanding of conversational AI architecture
- Hands-on experience with transformer-based models
- Implementation of secure banking operations
- Integration of multiple AI/ML technologies
- Development of production-ready applications

### Demonstration Points

- Multi-intent classification accuracy
- Entity extraction precision
- Dialogue state management
- Real-time analytics dashboards
- Security and compliance measures
- Scalable architecture design

## 📄 License

This project is licensed under the MIT License. You are free to use, modify, and distribute this project for educational and professional purposes.

## 🙏 Acknowledgments

- Hugging Face for transformer models
- Groq for LLM API services
- Streamlit for the amazing framework
- The open-source community for various libraries

## 🔮 Future Enhancements

- Multi-language support
- Voice interaction capabilities
- Mobile application integration
- Advanced fraud detection
- Blockchain integration for transactions
- Real-time push notifications
- Integration with actual banking APIs

---

**Note**: This is a demonstration project for educational and certification purposes. For production deployment, additional security measures and compliance requirements must be implemented.

*Built with Python, Streamlit, Transformers and Modern NLP Technologies*
