from output.exp import data
import argparse
import json
import os

from FlagEmbedding import FlagReranker
from llama_index.core import (
    QueryBundle,
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from tqdm import tqdm
from util.kg_post_processor import (
    GraphFilterPostProcessor,
    KGRetrievePostProcessor,
    NaivePostprocessor,
    ngram_overlap,
)


def kg_rag_parallel(
    questions,
    corpora,
    doc2kg,
    top_k=5,
    workers=4,
    persist_dir=None,
    reranker="../model/bge-reranker-large",
    dataset="hotpotqa",
    hops=1
):
    prediction = {"answer": {}, "sp": {}}

    doc_chunks = []
    chunks_index = dict()
    ents = set()
    for sample in corpora:
        for ctx in sample["context"]:
            ent = ctx[0]
            ents.add(ent)
            if ent not in chunks_index:
                chunks_index[ent] = dict()
            for i in range(len(ctx[1])):
                doc_chunk = TextNode(text=f"{ent}: {ctx[1][i]}", id_=f"{ent}##{str(i)}")
                doc_chunks.append(doc_chunk)
                if str(i) not in chunks_index[ent]:
                    chunks_index[ent][str(i)] = doc_chunk.text
    if (persist_dir is not None) and os.path.exists(persist_dir):
        print("Load index from persist dir")
        sc = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(sc)
    else:
        print("Create and save index to persist dir")
        index = VectorStoreIndex(doc_chunks, show_progress=True)
        if persist_dir is not None:
            os.makedirs(persist_dir, exist_ok=True)
            index.storage_context.persist(persist_dir=persist_dir)
    print(f"Index ready in persist dir {persist_dir}")
    retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)
    # Commented out answering parts - focusing on retrieval only
    # qa_rag_template_str = "Context information is below.\n{context_str}\nGive a short factoid answer (as few words as possible).\nQ: Were Scott Derrickson and Ed Wood of the same nationality?\nA: Yes.\nQ: Who was born earlier, Emma Bull or Virginia Woolf?\nA: Adeline Virginia Woolf.\nQ: The arena where the Lewiston Maineiacs played their home games can seat how many people?\nA: 3,677 seated.\nQ: What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?\nA: Chief of Protocol.\n---------------------\nQ: {query_str}\nA: "
    # qa_rag_prompt_template = PromptTemplate(qa_rag_template_str)
    # response_synthesizer = get_response_synthesizer(
    #     response_mode=ResponseMode.COMPACT, text_qa_template=qa_rag_prompt_template
    # )

    kg_post_processor1 = KGRetrievePostProcessor(
        dataset=dataset, ents=ents, doc2kg=doc2kg, chunks_index=chunks_index, hops=hops
    )
    bge_reranker = FlagReranker(model_name_or_path=reranker)
    kg_post_processor2 = GraphFilterPostProcessor(
        dataset=dataset,
        topk=top_k,
        use_tpt=False,  # Default value
        ents=ents,
        doc2kg=doc2kg,
        chunks_index=chunks_index,
        reranker=bge_reranker,
    )

    # Create a retriever with postprocessors for retrieval only
    # We'll directly use the retriever with postprocessors to get nodes
    postprocessors = [
        kg_post_processor1,
        kg_post_processor2,
        NaivePostprocessor(dataset=dataset),
    ]


    sps_count = []
    for sample in tqdm(questions):
        sample_id = sample["_id"]
        sample_question = sample["question"]

        # Direct retrieval with postprocessors (no answer generation)
        query_bundle = QueryBundle(query_str=sample_question)
        nodes = retriever.retrieve(sample_question)

        # Apply postprocessors with query bundle
        try:
            for postprocessor in postprocessors:
                nodes = postprocessor.postprocess_nodes(
                    nodes, query_bundle=query_bundle
                )
        except UnboundLocalError as e:
            print(f"Error in postprocessor: {e}")
            # If postprocessor fails, continue with original nodes
            # This ensures we still get some retrieval results even if KG processing fails
            pass

        # Extract supporting facts from retrieved nodes
        sps = [
            [
                node.id_.split("##")[0],
                int(node.id_.split("##")[1]),
            ]
            for node in nodes
        ]

        # Skip answer generation - use empty string or None
        prediction["answer"][sample_id] = (
            ""  # Empty answer since we're only doing retrieval
        )
        prediction["sp"][sample_id] = sps
        sps_count.append(len(sps))

    print(f"Avg #sps: {sum(sps_count) / len(sps_count)}")

    return prediction


def main(args):
    question_path = args.question_path
    corpus_path = args.corpus_path

    with open(question_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpora = json.load(f)

    ents = set()
    for sample in corpora:
        for ctx in sample["context"]:
            ents.add(ctx[0])

    kg_dir = args.kg_dir
    doc2kg = dict()
    print(f"\n{'-' * 20}\nLoading KGs")
    for ent in tqdm(ents):
        subkg_path = os.path.join(kg_dir, f"{ent.replace('/', '_')}.json")
        if not os.path.exists(subkg_path):
            continue
        with open(subkg_path, "r", encoding="utf=8") as fin:
            subkg = json.load(fin)
            if subkg and len(subkg.keys()) > 0:
                for seq in subkg.keys():
                    for triplet in subkg[seq]:
                        h, r, t = triplet
                        if (ngram_overlap(h, ent) >= 0.90) or (
                            ngram_overlap(ent, h) >= 0.90
                        ):
                            h = ent
                        if (ngram_overlap(t, ent) >= 0.90) or (
                            ngram_overlap(ent, t) >= 0.90
                        ):
                            t = ent
                        triplet = h, r, t
                    if len(subkg[seq]) == 0:
                        del subkg[seq]
                if len(subkg.keys()) > 0:
                    doc2kg[ent] = subkg

    # Commented out LLM initialization since we're only doing retrieval
    # model_name = args.model_name
    # print('Init Ollama model')
    # Settings.llm = Ollama(model=model_name,request_timeout=200)
    embed_model_name = args.embed_model_name
    print("Init Ollama embedding")
    Settings.embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
    top_k = args.top_k
    workers = args.num_workers
    persist_dir = args.persist_dir
    reranker = args.reranker
    prediction = kg_rag_parallel(
        questions,
        corpora,
        doc2kg,
        top_k=top_k,
        workers=workers,
        persist_dir=persist_dir,
        dataset=args.dataset,
        reranker=reranker,
        hops=args.hops
    )

    result_path = args.result_path
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(prediction, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="hotpotqa", help="Dataset name")
    # hotpot full
    parser.add_argument(
        "--question_path",
        type=str,
        default="../data/hotpotqa/hotpot_dev_distractor_v1.json",
        help="Path to the data file",
    )
    parser.add_argument(
        "--corpus_path",
        type=str,
        default="../data/hotpotqa/corpus.json",
        help="Path to the corpus file",
    )
    parser.add_argument(
        "--result_path",
        type=str,
        default="../output/hotpot/hotpot_dev_distractor_v1_full.json",
        help="Path to the result file",
    )
    parser.add_argument(
        "--kg_dir", type=str, default="../data/hotpotqa/kgs/extract_subkgs"
    )
    parser.add_argument(
        "--persist_dir",
        type=str,
        default="../data/ollama_index/hotpotqa",
        help="Directory to store the index",
    )
    parser.add_argument(
        "--embed_model_name",
        type=str,
        default="mixedbread-ai/mxbai-embed-large-v1",
        help="Ollama embedding model name for indexing",
    )
    parser.add_argument("--top_k", type=int, default=10, help="Top k similar documents")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for parallel processing",
    )
    parser.add_argument(
        "--reranker",
        type=str,
        default="BAAI/bge-reranker-large",
        help="Path of the reranker model",
    )
    parser.add_argument(
        "--hops",
        type=int,
        default=1,
        help="Number of hops for entity expansion in KGRetrievePostProcessor",
    )
    args = parser.parse_args()
    main(args)
