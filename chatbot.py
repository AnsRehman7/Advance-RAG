import re
from typing import Callable, Iterator
from langchain_core.documents import Document
import config
from llmClient import get_chat_model
from retriever import get_search_indexes, retrieve


ANSWER_PROMPT = """You are a helpful document assistant that answers questions about the user's documents.
The context below has numbered blocks: passages found for this question and a short overview of each file.

How to answer:
- Use only the facts in the context, but think about them like a person would.
- If the context answers the question, answer it directly and simply, even when the words differ a little.
  For example, "what is her favorite color?" is answered by "Her favorite colors are red and blue."
- Only when the user clearly asks for something more specific than the context gives, say what is missing
  and then give what the documents do say. For example, if asked for the one color she likes most and
  the context only lists two favorite colors, say that no single color is named as the top one and list both.
- If the question assumes something the context does not support, point that out politely.
- If the question is unclear, for example it says "he" or "it" and the chat history does not say
  who or what that is, ask a short question to clarify and say what the files are about.
- If different files say different things, give both and say which file says what.
- Never make up facts. You may draw a simple conclusion from the facts, but say that it is a conclusion.
- Only say that you could not find it in the documents when the context has nothing related to the question.

Citations:
- After each fact, add the number of the block it came from, like [2] or [1][3].
- Do not cite anything when you say that you could not find the answer.

Style:
- Reply in the same language and script as the question. If the user writes Urdu in English letters
  (Roman Urdu), reply in Roman Urdu.
- Match the length to the question: short for a simple question, more detail when the user asks for it.
- Use Markdown for lists and tables. Never use HTML tags like <br>.

Context:
{context}

Chat history:
{history}

Question: {question}
Answer:"""


ROUTE_PROMPT = """You decide how to handle a message sent to a chatbot that answers questions about the user's documents.
Reply with one word only: SEARCH, REWRITE or CHAT.

SEARCH: the user wants information on any topic or fact, even a general one, because the answer
must come from the documents. Examples: "summarize the book", "tell me about this book",
"what is chapter 2 about", "what is RAG?", "who is the author?", "tell me more about it".

REWRITE: the user wants the previous answer changed without new information, for example shorter,
simpler, translated or as a list. Examples: "make it shorter", "explain that in simple words",
"translate it to Urdu", "give me that as bullet points".

CHAT: a greeting, thanks or small talk, a question about the chatbot itself or whether it will help,
without asking for the content yet, or a question about which files are added. Examples: "hi",
"thanks", "who are you?", "what can you do?", "can you help me with my book?",
"will you answer from this book?", "so you will tell me what is written in this book according to my questions?",
"what files do you have?", "how many documents are uploaded?".

If you are not sure, reply SEARCH.

Chat history:
{history}

Message: {question}"""


CHAT_PROMPT = """You are a friendly document assistant that answers questions about the user's documents.
Only when asked who you are, say you are a document assistant. Do not say you are ChatGPT or name any AI model.
The message below does not need a search in the documents, so reply to it naturally and briefly.
If it fits, tell the user what you can do and invite them to ask a question about their files.
If asked which files are added, list them with one line about each from its overview.
Do not state other facts about the content of the files.
Reply in the same language and script as the message, for example Roman Urdu if the user writes in it.

The user's files:
{files}

Chat history:
{history}

Message: {question}
Reply:"""


REWRITE_PROMPT = """You are a document assistant. The user wants your previous answer changed,
for example shorter, simpler, translated or as a list. Rewrite it the way they ask.
Use only the facts in the chat history and do not add new ones.
Reply in the language the user asks for, otherwise in the same language and script as the message.
Use Markdown for lists and tables. Never use HTML tags like <br>.

Chat history:
{history}

Message: {question}
Reply:"""


def route(question: str, history: str) -> str:
    """Ask the LLM if the message needs a search (SEARCH), a change of the last answer (REWRITE) or just a reply (CHAT)."""
    prompt = ROUTE_PROMPT.format(history=history, question=question)
    decision = get_chat_model().invoke(prompt).content.strip().upper()

    for name in ("CHAT", "REWRITE"):
        if name in decision:
            return name

    # when unsure, search, so a real question is never left without an answer
    return "SEARCH"


def format_history(messages: list[dict]) -> str:
    """Turn the last few chat messages into plain text for the prompt."""
    if config.MEMORY_WINDOW <= 0 or not messages:
        return "None"

    # one turn = one user message + one assistant message
    recent = messages[-config.MEMORY_WINDOW * 2:]

    lines = []
    for m in recent:
        # old citation numbers like [2] point to old context blocks, so remove them
        content = re.sub(r" ?\[\d+\]", "", m["content"])
        lines.append(f"{m['role']}: {content}")
    return "\n".join(lines)


def source_label(chunk: Document) -> str:
    source = chunk.metadata.get("source", "unknown")
    page = chunk.metadata.get("page")

    # PyPDFLoader counts pages from 0
    if page is not None:
        return f"{source} (page {page + 1})"
    return source


def build_context(chunks: list[Document], summaries: dict):
    """
    Number the found chunks and the file overviews as blocks [1], [2], ... so the LLM can cite them.
    Chunks are added until the text gets longer than MAX_CONTEXT_CHARS, the short overviews are always added.

    Returns the context text and the source label of each block number.
    """
    parts = []
    labels = {}
    total_chars = 0

    for chunk in chunks:
        number = len(parts) + 1
        part = f"[{number}] {source_label(chunk)}\n{chunk.page_content}"
        if total_chars + len(part) > config.MAX_CONTEXT_CHARS:
            break
        parts.append(part)
        labels[number] = source_label(chunk)
        total_chars += len(part)

    for name, summary in summaries.items():
        number = len(parts) + 1
        parts.append(f"[{number}] {name} (overview)\n{summary}")
        labels[number] = f"{name} (overview)"

    return "\n\n".join(parts) or "None", labels


def cited_sources(answer: str, labels: dict) -> list[str]:
    """List the blocks the answer cites, like "[2] self.txt", so only the sources really used are shown."""
    numbers = set()
    for group in re.findall(r"\[([\d,\s]+)\]", answer):
        numbers.update(int(n) for n in re.findall(r"\d+", group))

    return [f"[{n}] {labels[n]}" for n in sorted(numbers) if n in labels]


def stream_text(prompt: str) -> Iterator[str]:
    """Send the prompt to the LLM and yield the answer piece by piece as it is written."""
    for chunk in get_chat_model().stream(prompt):
        if chunk.content:
            # some models cite with 【1】 instead of [1]
            yield chunk.content.replace("【", "[").replace("】", "]")


def answer_question(question: str, messages: list[dict]) -> tuple[Iterator[str], Callable[[str], list[str]]]:
    """
    Decide what the message needs and start the reply.

    Returns a stream of the answer text, and a function that takes the finished answer
    and returns its sources (the answer must be complete to know which blocks it cites).
    """
    history = format_history(messages)
    _, doc_store, _ = get_search_indexes()
    files = doc_store["files"] if doc_store else []
    summaries = doc_store["summaries"] if doc_store else {}

    decision = route(question, history)

    if decision == "CHAT":
        file_list = "\n".join(f"- {name}: {summaries.get(name, 'no overview yet')}" for name in files) or "None"
        prompt = CHAT_PROMPT.format(files=file_list, history=history, question=question)
        return stream_text(prompt), lambda answer: []

    if decision == "REWRITE" and messages:
        # the facts stay the same, so the sources of the previous answer still apply
        previous_sources = messages[-1].get("sources", [])
        prompt = REWRITE_PROMPT.format(history=history, question=question)
        return stream_text(prompt), lambda answer: previous_sources

    chunks = retrieve(question, history)
    context, labels = build_context(chunks, summaries)
    prompt = ANSWER_PROMPT.format(context=context, history=history, question=question)
    return stream_text(prompt), lambda answer: cited_sources(answer, labels)
