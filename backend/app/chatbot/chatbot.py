from app.chatbot.chatbot_utils import *

import ast
import regex as re
import argparse
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.documents import Document
from typing_extensions import List, TypedDict, Dict
from langchain.prompts import ChatPromptTemplate
from langgraph.types import interrupt, Command
from langgraph.checkpoint.mongodb import MongoDBSaver

from groq import Groq
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from pymongo import MongoClient

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    desired_fact: Dict[str, List[str]]
    fact_provided: Dict[str, str]

def identify_facts(llm, query):
    query_result = llm.chat.completions.create(
        messages=[{
            "role": "system",
            "content": 
            """
                You will be given a prompt from the user containing questions about drug facts or medical related object.
                Your job is to identify facts that can help determine the object the user is referring to, your job is not to answer the question.
                The types of informations and their explanations are:
                1. Object Name: The name of the object as listed on the site (eg: Emturnas Drops 15 ml).
                2. Instructions: Instructions on when and how the object should be used (eg: After meals).
                3. Dosage: Information on the recommended dosage or amount of consumption, can be based on age or condition.
                4. Side Effects: Side effects that may arise after taking the object.
                5. Category: Legal category of the drug, such as: Over-the-Counter Drugs — can be purchased without a prescription, Limited Over-the-Counter Drugs — can be purchased freely with certain restrictions, Prescription Drugs — can only be purchased with a prescription, Consumer Products — not prescription drugs (such as mild itch ointments, etc.).
                6. General Indications: General uses of the object, namely to treat certain symptoms or diseases.
                7. Shape and size: The shape and size of the product packaging (eg: Box, Bottle @ 15 ml).
                8. Composition: The content or active substance in the object.
                9. Contraindications: Situations or conditions that prevent the object from being used (eg: severe liver dysfunction).
                10. Manufacturer: The name of the company or factory that produces the object.
                11. Warning: Special warnings before using this object, such as prohibitions on use in certain conditions.
                12. Description: A brief explanation of the object in general, often including the purpose and how the drug works.

                Example:
                Apa efek samping, aturan pakai, dan siapa yang membuat obat untuk meredakan demam yang bernama panadol.

                Provide the same analysis steps as the steps below:
                1. Translate the prompt to english. In this case 'What are the side effects, instructions, and who makes the drug to relieve fever called panadol.'
                2. Identify the type of informations desired by the user and explain your reasoning. if there is no information desired by the user then keep it empty. In this case, because the user asks for side effects, instructions, and who makes it, the type of informations desired is [Side Effects, Instructions, Manufacturer].
                3. Identify the information provided by the user that can help identify the object the user is referring to complete it with a verb. In this case, because the user mentioned that the medicine can relieve fever and is called panadol, the information that can help identify the medicine is [medicine to relieve fever, medicine called panadol]
                4. Determine the type of informations and explain your reasoning from the information that has been identified by looking at the explanation of the 12 types of information if the information doesn't fit with the 12 mentioned then don't list it. Make sure the user is sure of the informations and the user's intention is not to confirm the informations. Because the information 'medicine to relieve fever' is the general use of the medicine and the information 'medicine called panadol' is the name of the medicine, the type of each information is [to relieve fever: General Indications, panadol: Drug Name].
                5. Create a dictionary that contains the type of fact the user wants, the type of fact and the fact. In this case {'Desired fact': ['Side Effects', 'Instructions', 'Manufacturer'], 'Fact provided': {'General Indications': 'to relieve fever', 'Object Name': 'panadol'}}
                6. Translate the information (not the type) provided by the user into indonesian again but not the desired fact. In this case {'Desired fact': ['Side Effects', 'Instructions', 'Manufacturer'], 'Fact provided': {'General Indications': 'untuk meredakan demam', 'Object Name': 'panadol'}}
                7. Remember not to include notes or additions to the output, it must remain a dictionary.

                Output: {'Desired fact': ['Side Effects', 'Instructions', 'Manufacturer'], 'Fact provided': {'General Indications': 'untuk meredakan demam', 'Object Name': 'panadol'}}
                Final output format: JSON-style dictionary as above.
                Do not answer the user's question, just identify the informations.
            """
        }, {
            "role": "user",
            "content": query
        }],
        model="llama3-8b-8192",
        temperature=0,
    )

    answer = query_result.choices[0].message.content
    print(answer)
    answer = re.findall(r"\{.*?\}(?=(?:\n|$))", answer, re.DOTALL)[-1]
    answer = ast.literal_eval(answer)

    desired_fact = answer["Desired fact"]
    fact_provided = answer["Fact provided"]

    dct = {
        "Object Name": "Nama Obat",
        "Instructions": "Aturan Pakai",
        "Dosage": "Dosis",
        "Side Effects": "Efek Samping",
        "Category": "Golongan Produk",
        "General Indications": "Indikasi Umum",
        "Shape and size": "Kemasan",
        "Composition": "Komposisi",
        "Contraindications": "Kontra Indikasi",
        "Manufacturer": "Manufaktur",
        "Warning": "Perhatian",
        "Description": "Deskripsi"
    }

    fact_provided = {dct[fact_type]: fact for fact_type, fact in fact_provided.items() if fact_type in dct.keys()}

    return desired_fact, fact_provided

def hybrid_retrieve(df, lexical_retrievers, semantic_retriever, desired_fact, fact_provided, k):
    jaro_winkler_ranking = JaroWinklerRanking(df)
    lexical_ranking = LexicalRanking(lexical_retrievers, df)
    semantic_ranking = SemanticRanking(semantic_retriever, df)

    jaro_winkler_rank = jaro_winkler_ranking.rank(fact_provided)
    lexical_rank = lexical_ranking.rank(fact_provided)
    semantic_rank = semantic_ranking.rank(fact_provided)

    hybird_rank = pd.merge(left=lexical_rank, right=semantic_rank, how="inner", on="id")
    hybird_rank = pd.merge(left=hybird_rank, right=jaro_winkler_rank, how="inner", on="id")
    hybird_rank = ReciprocalRankFusion(hybird_rank, 60, "hybird_rank")

    retrieved_docs = df.loc[hybird_rank.sort_values(by="hybird_rank")["id"].to_list()[0:k]]

    retrieved_docs = [
        Document(
            page_content="\n".join(f"{col_name}: {value}" for col_name, value in row.items() 
                                   if col_name not in ["Link Obat", "Check", "Link Gambar"]),
            metadata={"row_index": idx}
        )
        for idx, row in retrieved_docs.iterrows()
    ]

    return retrieved_docs

def _no_fact_(state: State, config: dict):
    question = interrupt("Adakah deskripsi yang dapat membantu saya untuk mencari obat yang ada maksud?")
    return {"question": question}

def _identify_facts_(state: State, config: dict):
    query_llm = config["configurable"]["query_llm"]
    desired_fact, fact_provided = identify_facts(query_llm, state["question"])
    return {"desired_fact": desired_fact, "fact_provided": fact_provided}

def _retrieve_or_ask_again_(state: State, config: dict):
    if len(state["fact_provided"]) == 0:
        return "no_fact"
    else:
        return "retrieve"

def _retrieve_(state: State, config: dict):
    df = config["configurable"]["df"]
    lexical_retrievers = config["configurable"]["lexical_retrievers"]
    semantic_retriever = config["configurable"]["semantic_retriever"]
    desired_fact = state["desired_fact"]
    fact_provided = state["fact_provided"]
    retrieved_docs = hybrid_retrieve(df, lexical_retrievers, semantic_retriever, desired_fact, fact_provided, 10)
    return {"context": retrieved_docs}


def _generate_(state: State, config: dict):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the provided context to answer the user's question and answer in Bahasa. At the end of the answer ask the user if they're satisfied with the answer."),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])
    messages = prompt.invoke({
        "question": state["question"],
        "context": docs_content
    })
    response = config["configurable"]["llm"].invoke(messages)
    return {"answer": response.content}

from langgraph.graph import START, StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import uuid

graph_builder = StateGraph(State)

graph_builder.add_node("identify_fact", _identify_facts_)
graph_builder.add_node("no_fact", _no_fact_)
graph_builder.add_node("retrieve", _retrieve_)
graph_builder.add_node("generate", _generate_)
graph_builder.add_edge(START, "identify_fact")
graph_builder.add_conditional_edges("identify_fact", _retrieve_or_ask_again_, {"retrieve": "retrieve", "no_fact": "no_fact"})
graph_builder.add_edge("no_fact", "identify_fact")
graph_builder.add_edge("retrieve", "generate")
graph_builder.add_edge("generate", END)
client = MongoClient("mongodb://localhost:27017")
db = client["langgraph"]
collection = db["checkpoints"]
checkpointer = MongoDBSaver(client=client, collection=collection)
graph = graph_builder.compile(checkpointer=checkpointer)

def start_qa(question, graph, config):
    result = graph.invoke({"question": question}, config=config)
    return result

def resume_qa(question, graph, config):
    result = graph.invoke(Command(resume=question), config=config)
    return result

def init_components():
    load_dotenv()
    df = pd.read_csv("./app/chatbot/scrapping_auto_df.csv")
    col_to_embed = [
        "Aturan Pakai", "Dosis", "Efek Samping", "Golongan Produk", "Indikasi Umum",
        "Kemasan", "Komposisi", "Kontra Indikasi", "Perhatian", "Deskripsi"
    ]
    create_retriever = CreateRetriever(df, col_to_embed)
    lexical_retrievers = create_retriever.create_lexical_retriever()
    semantic_retriever = create_retriever.create_semantic_retriever("./app/chatbot/halodoc_db", "intfloat/multilingual-e5-large-instruct")
    # semantic_retriever = create_retriever.create_semantic_retriever("./app/chatbot/halodoc_db", "./app/chatbot/embedding_model/e5")
    query_llm = Groq(api_key=os.getenv("GROQ_KEY"))
    llm = ChatGroq(model="llama3-8b-8192", temperature=0, api_key=os.getenv("GROQ_KEY"))
    return df, lexical_retrievers, semantic_retriever, query_llm, llm


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--query", help="pertanyaan untuk bot")
    args = parser.parse_args()

    if args.query:
        query = args.query

        df, lexical_retrievers, semantic_retriever, query_llm, llm = init_components()

        state = {
            "df": df,
            "lexical_retrievers": lexical_retrievers,
            "semantic_retriever": semantic_retriever,
            "query_llm": query_llm,
            "question": query,
            "llm": llm
        }

        result = graph.invoke(state)
        print(f'Context: {result["context"]}\\n\\n')
        print(f'Answer: {result["answer"]}')
    else:
        print("Input query")
