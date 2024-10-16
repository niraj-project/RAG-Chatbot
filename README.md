# **Cybersecurity RAG Chatbot**

This project implements a Retrieval Augmented Generation (RAG) chatbot that provides users with cybersecurity best practices and recommendations. It combines Pinecone vector database for document retrieval, Sentence Transformers for query embedding, and a conversational chatbot powered by Claude v1 API. The chatbot retrieves relevant documents based on user queries and uses them to generate contextually accurate responses.

## **Features**
- **Document Retrieval**: Retrieves relevant documents using Pinecone.
- **Natural Language Understanding**: Uses Claude v1 API to generate conversational responses.
- **Cybersecurity Focus**: Designed to answer queries related to cybersecurity best practices.

---

## Technologies Used
- Backend: Python (3.8+)
- Web Server: Flask
- Vector Database: Pinecone
- Text Embeddings: Sentence Transformers
- Conversation Generation: Claude v1 API
- Frontend: HTML/CSS


## **1. Prerequisites**

Before starting the project, ensure you have the following installed:

- **Python 3.8+**
- **An IDE or text editor** (e.g., Visual Studio Code)
- **Pinecone API key** (for vector database)
- **Claude v1 API key** (for chatbot responses)

  
## **2. Install Required Libraries**

Create and activate a virtual environment (optional but recommended):

```bash
# On Linux/MacOS
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate

```

## **3.Set Up Environment Variables**

Make sure to add your own API keys in the code. Open the Python file and insert the keys for Pinecone and Claude v1 API where required.
**Python**
```bash
PINECONE_API_KEY = "your-pinecone-api-key"
API_KEY = 'your-claude-v1-api-key'
```

## **4.Start the Flask Application**
Run the Flask server locally by executing the following command:
```
python app.py
```
The application should now be running locally on http://127.0.0.1:5000/.

## **5.Access the Chatbot**
Open your browser and go to http://localhost:5000 or http://127.0.0.1:5000. You should see the chatbot interface where you can type in cybersecurity-related questions and get responses from the chatbot.

---

## **Folder Structure**
```
cybersecurity-rag-chatbot/
├── app.py                  # Main Flask application
├── templates/
│   └── index.html          # HTML file for the chatbot UI
├── static/
│   └── styles.css          # CSS file for styling the UI
├── requirements.txt        # Python dependencies
├── README.md               # Project overview and setup instructions
└── error_log.log           # Log file for any errors
```
## **API Keys**
- **Pinecone:** To create a Pinecone account and get an API key, visit Pinecone.io.
- **Claude v1 API:** You can get a key from OpenRouter or any provider offering Claude API access.

## **Logs**
The project logs errors and API responses to error_log.log. If you encounter issues, you can inspect this log file for detailed information.

## **Contributing**
Feel free to submit issues or pull requests if you find bugs or want to contribute to the project. You can also extend the chatbot's abilities by integrating other document sources or improving its response generation.

## **License**
This project is licensed under the MIT License - see the LICENSE file for details.
