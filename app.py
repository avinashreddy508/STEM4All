import streamlit as st
import configparser
from snowflake.snowpark import Session
from snowflake.connector.errors import ProgrammingError
from snowflake.snowpark.exceptions import SnowparkSessionException, SnowparkSQLException
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

slide_window = 3  # How many conversations to remember for context
num_chunks = 3  # Number of chunks to provide as context

model = SentenceTransformer('all-MiniLM-L6-v2')


def initialize_messages():
    """Initialize or reset chat history."""
    if st.session_state.clear_conversation or "messages" not in st.session_state:
        st.session_state.messages = []


def configure_options():
    """Sidebar configuration options for model and session management."""
    st.sidebar.selectbox(
        'Select your model:', ['mistral-large2', 'mixtral-8x7b', 'mistral-large'], key="model_name"
    )
    st.sidebar.checkbox('Remember chat history?', key="use_chat_history", value=True)
    st.sidebar.checkbox('Debug: Show summary of previous conversations', key="debug", value=True)
    st.sidebar.button("Start Over", key="clear_conversation")
    st.sidebar.expander("Session State").write(st.session_state)


def get_chat_history():
    """Retrieve the chat history based on the defined slide window."""
    # Get the history from the st.session_stage.messages according to the slide window parameter

    chat_history = []

    start_index = max(0, len(st.session_state.messages) - slide_window)
    for i in range(start_index, len(st.session_state.messages) - 1):
        chat_history.append(st.session_state.messages[i])

    return chat_history


def create_session():
    """Create a Snowflake session using configuration details."""
    config = configparser.ConfigParser()
    config.read('config/properties.ini')
    snowflake_config = config['Snowflake']

    connection_params = {key: snowflake_config.get(key) for key in
                         ['account', 'user', 'password', 'role', 'warehouse', 'database', 'schema']}
    session = Session.builder.configs(connection_params).create()
    return session


def query_available_docs(session):
    """Query Snowflake to list available documents from a stage."""
    return session.sql("ls @mystage").collect()


def complete_response(session, question, rag):
    session = create_session()

    prompt, url_link, relative_path = create_prompt(session, question, rag)

    response = session.sql("SELECT snowflake.cortex.complete(?, ?),?",
                           params=[st.session_state.model_name, prompt, json.dumps({'guardrails': True})]).collect()
    return response[0]['SNOWFLAKE.CORTEX.COMPLETE(?, ?)'], url_link, relative_path


def summarize_question_with_history(session, chat_history, question):
    # To get the right context, use the LLM to first summarize the previous conversation
    # This will be used to get embeddings and find similar chunks in the docs for context

    prompt = f"""
        You are an expert chat assistance that extracts information and 
        refine the question based on the provided chat history and context.
           You offer a chat experience considering the information included in the CHAT HISTORY
           provided between <chat_history> and </chat_history> tags.
           When answering the question contained between <question> and </question> tags,
           be concise and do not hallucinate. 
           If you don’t have the information, just say so.

           Do not mention the CONTEXT used in your answer.
           Do not mention the CHAT HISTORY used in your answer.

           Only answer the question if you can extract it from the CONTEXT provided.
        You are an expert assistant. Refine the question based on the provided chat history and context.
        <chat_history>
        {chat_history}
        </chat_history>
        <question>
        {question}
        </question>
        """
    # cmd = """
    #            select snowflake.cortex.complete(?, ?),? as response
    #
    #        """
    # df_response = session.sql(cmd, params=[st.session_state.model_name,  prompt, json.dumps({'guardrails': True})]).collect()

    response = session.sql("SELECT snowflake.cortex.complete(?, ?),?",
                           params=[st.session_state.model_name, prompt, json.dumps({'guardrails': True})]).collect()
    df_response = response[0]['SNOWFLAKE.CORTEX.COMPLETE(?, ?)']
    sumary = df_response

    if st.session_state.debug:
        st.sidebar.text("Summary to be used to find similar chunks in the docs:")
        st.sidebar.caption(sumary)

    sumary = sumary.replace("'", "")
    return sumary


def create_prompt(session, question, rag):
    global prompt_context, relative_path
    if st.session_state.use_chat_history:
        chat_history = get_chat_history()

        if chat_history:  # There is chat history
            question_summary = summarize_question_with_history(session, chat_history, question)
            if rag:
                prompt_context, relative_path = get_similar_chunks(session, question_summary)
            else:
                prompt_context = None
        else:  # First question when using chat history
            if rag:
                prompt_context, relative_path = get_similar_chunks(session, question)
            else:
                prompt_context = None
    else:
        chat_history = ""
        if rag:
            prompt_context = get_similar_chunks(session, question)
        else:
            prompt_context = None

    if prompt_context:  # Using retrieved context
        prompt = f"""
           You are an expert chat assistance that extracts information from the CONTEXT provided
           between <context> and </context> tags.
           You offer a chat experience considering the information included in the CHAT HISTORY
           provided between <chat_history> and </chat_history> tags.
           When answering the question contained between <question> and </question> tags,
           be concise and do not hallucinate. 
           If you don’t have the information, just say so.

           Do not mention the CONTEXT used in your answer.
           Do not mention the CHAT HISTORY used in your answer.

           Only answer the question if you can extract it from the CONTEXT provided.

           <chat_history>
           {chat_history}
           </chat_history>
           <context>          
           {prompt_context}
           </context>
           <question>  
           {question}
           </question>
           Answer: 
        """

    #
    else:  # Default fallback case
        if chat_history:  # There is chat history
            question_summary = summarize_question_with_history(session, chat_history, question)
            prompt = f"""'Question:
                          {question_summary}
                          Answer: '
                       """
            relative_path = "None"
        else:
            prompt = f"""'Question:
               {question}
               Answer: '
            """
            relative_path = "None"

    return prompt, None, relative_path


def get_similar_chunks(session, question):
    """Retrieve the most similar document chunks for the provided question."""
    query = """
        WITH results AS (
            SELECT RELATIVE_PATH,
                   VECTOR_COSINE_SIMILARITY(docs_chunks_table.chunk_vec, 
                   SNOWFLAKE.CORTEX.EMBED_TEXT_768('e5-base-v2', ?)) AS similarity,
                   chunk
            FROM docs_chunks_table
            ORDER BY similarity DESC
            LIMIT ?
        )
        SELECT chunk, relative_path FROM results
    """
    df_chunks = session.sql(query, params=[question, num_chunks]).to_pandas()
    similar_chunks = "".join([df_chunks._get_value(i, 'CHUNK') for i in range(len(df_chunks) - 1)])
    return similar_chunks.replace("'", ""), df_chunks._get_value(0, 'RELATIVE_PATH')


st.set_page_config(
    page_title="STEM4All App",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# Welcome to Inclusion! A powerful app for STEM4All."
    }
)

st.sidebar.image("STEM4All_Empowering_Minds.jpeg", caption="STEM4All Chat Assistant", use_column_width=True)


# Main function for the Streamlit app
def main():
    st.title(
        f":speech_balloon: Inclusion: Your 24/7 Companion for STEM4All Empowering Every Mind to Engineer the Future of Data,")

    # Initialize messages in session state if not already done
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    try:
        session = create_session()

    except (ProgrammingError, SnowparkSQLException, SnowparkSessionException) as e:
        if "warehouse is suspended" in str(e):
            st.stop()  # Stop execution if the warehouse is unavailable
        else:
            st.error("An unexpected error occurred: " + str(e))
            st.stop()

    # Sidebar settings for emotion threshold and RAG toggle
    with st.sidebar:
        st.write("Configuration Settings")
        rag = st.checkbox('Use your own documents as context?')

    # Expander to display available documents
    with st.expander("Show available documents"):
        st.write("Available documents:")
        available_docs = query_available_docs(session)
        st.dataframe([doc["name"] for doc in available_docs])

    # Initialize sidebar options and chat messages
    configure_options()
    initialize_messages()

    # Display chat messages in session
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if user_question := st.chat_input("What would you like to discuss today? (e.g., Snowflake, Data Warehouse)"):
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # Display assistant's response with a loading spinner
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            sanitized_question = user_question.replace("'", "")
            with st.spinner(f"{st.session_state.model_name} is thinking..."):
                # try:
                response, url_link, relative_path = complete_response(session, sanitized_question, rag)
                if response:  # Check if response is not None
                    message_placeholder.markdown(response)

                    if rag:
                        st.markdown(f"Link to [{relative_path}]({url_link})")
                    # Store the assistant's response in the session history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    message_placeholder.markdown("No response received. Please try again.")


if __name__ == "__main__":
    main()

