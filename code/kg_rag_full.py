import argparse
import json
import os

from FlagEmbedding import FlagReranker

# from llama_index.llms.ollama import Ollama
from llama_index.core import (
    Settings,
    # PromptTemplate,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import QueryBundle, TextNode

# from llama_index.core.query_engine import RetrieverQueryEngine
# from llama_index.core.response_synthesizers import ResponseMode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from tqdm import tqdm
from util.kg_post_processor import (
    GraphFilterPostProcessor,
    KGRetrievePostProcessor,
    NaivePostprocessor,
    ngram_overlap,
)

# from util.kg_response_synthesizer import get_response_synthesizer


def kg_rag_parallel(
    data,
    doc2kg,
    top_k=5,
    workers=4,
    persist_dir=None,
    dataset=None,
    reranker="BAAI/bge-reranker-large",
):
    prediction = {"answer": {}, "sp": {}}

    doc_chunks = []
    chunks_index = dict()
    ents = set()
    sample_idx = 0
    for sample in data:
        for ctx in sample["context"]:
            ent = ctx[0]
            ents.add(ent)
            if ent not in chunks_index:
                chunks_index[ent] = dict()
            for i in range(len(ctx[1])):
                # MuSiQue expects idx##entity##seq format, HotpotQA expects entity##seq
                if dataset and dataset.lower() == "musique":
                    node_id = f"{sample_idx}##{ent}##{str(i)}"
                else:
                    node_id = f"{ent}##{str(i)}"
                doc_chunk = TextNode(text=f"{ent}: {ctx[1][i]}", id_=node_id)
                doc_chunks.append(doc_chunk)
                if str(i) not in chunks_index[ent]:
                    chunks_index[ent][str(i)] = doc_chunk.text
        sample_idx += 1
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
    # qa_rag_template_str = "Context information is below.\n{context_str}\nGive a short factoid answer (as few words as possible).\nQ: Were Scott Derrickson and Ed Wood of the same nationality?\nA: Yes.\nQ: Who was born earlier, Emma Bull or Virginia Woolf?\nA: Adeline Virginia Woolf.\nQ: The arena where the Lewiston Maineiacs played their home games can seat how many people?\nA: 3,677 seated.\nQ: What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?\nA: Chief of Protocol.\n---------------------\nQ: {query_str}\nA: "
    # qa_rag_prompt_template = PromptTemplate(qa_rag_template_str)
    # response_synthesizer = get_response_synthesizer(
    #     response_mode=ResponseMode.COMPACT, text_qa_template=qa_rag_prompt_template
    # )

    kg_post_processor1 = KGRetrievePostProcessor(
        dataset=dataset, ents=ents, doc2kg=doc2kg, chunks_index=chunks_index
    )
    bge_reranker = FlagReranker(model_name_or_path=reranker)
    kg_post_processor2 = GraphFilterPostProcessor(
        dataset=dataset,
        topk=top_k,
        use_tpt=False,
        ents=ents,
        doc2kg=doc2kg,
        chunks_index=chunks_index,
        reranker=bge_reranker,
    )

    # engine = RetrieverQueryEngine(
    #     retriever=retriever,
    #     response_synthesizer=response_synthesizer,
    #     node_postprocessors=[
    #         kg_post_processor1,
    #         kg_post_processor2,
    #         NaivePostprocessor(),
    #     ],
    # )

    test_size = len(data)

    sps_count = []
    node_postprocessors = [
        kg_post_processor1,
        kg_post_processor2,
        NaivePostprocessor(dataset=dataset),
    ]
    for sample in tqdm(data[: min(len(data), test_size)]):
        sample_id = sample["_id"]
        sample_question = sample["question"]
        # sample_answer = sample["answer"]  # Not needed for retrieval-only benchmark

        # Retrieval only - no answer generation
        retrieved_nodes = retriever.retrieve(sample_question)
        # Apply post-processors manually
        query_bundle = QueryBundle(query_str=sample_question)
        for postprocessor in node_postprocessors:
            retrieved_nodes = postprocessor.postprocess_nodes(
                retrieved_nodes, query_bundle
            )

        # response = engine.query(sample_question)
        # answer = response.response
        answer = ""  # Placeholder - no answer generation

        # Parse supporting facts based on dataset format
        sps = []
        for source_node in retrieved_nodes:
            parts = source_node.node.id_.split("##")
            if dataset and dataset.lower() == "musique":
                # MuSiQue format: idx##entity##seq
                entity = parts[1]
                seq = int(parts[2])
            else:
                # HotpotQA format: entity##seq
                entity = parts[0]
                seq = int(parts[1])
            sps.append([entity, seq])

        prediction["answer"][sample_id] = answer
        prediction["sp"][sample_id] = sps
        sps_count.append(len(sps))

    print(f"Avg #sps: {sum(sps_count) / len(sps_count)}")

    return prediction


def normalize_triviaqa_data(data):
    """Convert TriviaQA format to HotpotQA format for compatibility."""
    normalized = []
    for sample in data:
        normalized_sample = {
            "_id": sample.get("_id", sample.get("question_id", "")),
            "question": sample["question"],
            "answer": sample["answer"],
            "context": [],
        }

        # Handle context - TriviaQA has [title, [text1, text2, ...]] format
        if "context" in sample:
            for ctx in sample["context"]:
                if len(ctx) >= 2:
                    title = ctx[0]
                    texts = ctx[1]
                    # Ensure texts is a list
                    if isinstance(texts, str):
                        texts = [texts]
                    elif not isinstance(texts, list):
                        texts = [str(texts)]

                    # Clean up texts - remove empty strings and trim
                    texts = [text.strip() for text in texts if text and text.strip()]
                    if texts:  # Only add if we have valid texts
                        normalized_sample["context"].append([title, texts])

        # Add empty supporting facts if not present (TriviaQA doesn't have them)
        if "supporting_facts" not in sample:
            normalized_sample["supporting_facts"] = []
        else:
            normalized_sample["supporting_facts"] = sample["supporting_facts"]

        normalized.append(normalized_sample)
    return normalized


def normalize_musique_data(data):
    """Convert MuSiQue format to HotpotQA format for compatibility."""
    normalized = []
    for sample in data:
        normalized_sample = {
            "_id": sample.get(
                "id", sample.get("_id", "")
            ),  # MuSiQue uses "id", HotpotQA uses "_id"
            "question": sample["question"],
            "answer": sample["answer"],
            "context": [],
        }
        # Convert paragraphs to context format
        for para in sample["paragraphs"]:
            # MuSiQue: [title, [text]]
            # HotpotQA: [title, [sent1, sent2, ...]]
            title = para["title"]
            text = para["paragraph_text"]
            # Split text into sentences (simple split by period for now)
            sentences = [s.strip() + "." for s in text.split(".") if s.strip()]
            normalized_sample["context"].append([title, sentences])
        normalized.append(normalized_sample)
    return normalized


def main(args):
    data_path = args.data_path
    original_data = None  # Keep original data for MuSiQue/TriviaQA output format

    # Load data based on dataset format
    with open(data_path, "r", encoding="utf-8") as f:
        if args.dataset.lower() == "musique":
            # MuSiQue uses JSONL format (one JSON object per line)
            original_data = [json.loads(line) for line in f if line.strip()]
            # Normalize MuSiQue format to HotpotQA format
            data = normalize_musique_data(original_data)
        elif args.dataset.lower() == "triviaqa":
            # TriviaQA uses JSON array format but needs normalization
            original_data = json.load(f)
            # Normalize TriviaQA format to HotpotQA format
            data = normalize_triviaqa_data(original_data)
        else:
            # HotpotQA uses JSON array format
            data = json.load(f)

    ents = set()
    for sample in data:
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

    # model_name = args.model_name
    # print("Init Ollama model")
    # Settings.llm = Ollama(model=model_name, request_timeout=200)
    embed_model_name = args.embed_model_name
    print("Init Ollama embedding")
    Settings.embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
    top_k = args.top_k
    workers = args.num_workers
    persist_dir = args.persist_dir
    reranker = args.reranker
    prediction = kg_rag_parallel(
        data,
        doc2kg,
        top_k=top_k,
        workers=workers,
        persist_dir=persist_dir,
        dataset=args.dataset,
        reranker=reranker,
    )

    result_path = args.result_path
    result_dir = os.path.dirname(result_path)
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)

    # Output format depends on dataset
    if args.dataset.lower() == "musique" and original_data is not None:
        # MuSiQue format: JSONL with one prediction per line
        # Need to map titles back to paragraph indices

        with open(result_path, "w", encoding="utf-8") as f:
            for sample_id in prediction["answer"].keys():
                # Get supporting paragraph titles
                sp_data = prediction["sp"].get(sample_id, [])
                sp_titles = [sp[0] for sp in sp_data]

                # Find the original sample to get paragraph indices
                original_sample = None
                for sample in original_data:
                    if sample["id"] == sample_id:
                        original_sample = sample
                        break

                # Map titles to paragraph indices
                predicted_support_idxs = []
                if original_sample:
                    title_to_idx = {}
                    for para in original_sample["paragraphs"]:
                        title_to_idx[para["title"]] = para["idx"]

                    seen_idxs = set()
                    for title in sp_titles:
                        if (
                            title in title_to_idx
                            and title_to_idx[title] not in seen_idxs
                        ):
                            predicted_support_idxs.append(title_to_idx[title])
                            seen_idxs.add(title_to_idx[title])

                pred_instance = {
                    "id": sample_id,
                    "predicted_answer": prediction["answer"].get(sample_id, ""),
                    "predicted_support_idxs": predicted_support_idxs,
                    "predicted_answerable": True,
                }
                f.write(json.dumps(pred_instance) + "\n")
    elif args.dataset.lower() == "triviaqa" and original_data is not None:
        # TriviaQA format: JSON array with additional fields
        triviaqa_prediction = {"answer": {}, "sp": {}}

        for sample_id in prediction["answer"].keys():
            # For TriviaQA, we need to map back to the original question_id
            original_sample = None
            for sample in original_data:
                if sample.get("_id") == sample_id or sample.get("question_id") == sample_id:
                    original_sample = sample
                    break

            if original_sample:
                original_id = original_sample.get("question_id", original_sample.get("_id", sample_id))
                triviaqa_prediction["answer"][original_id] = prediction["answer"][sample_id]
                triviaqa_prediction["sp"][original_id] = prediction["sp"][sample_id]

        # Include original data metadata if needed
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(triviaqa_prediction, f, indent=2)
    else:
        # HotpotQA format: single JSON object
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(prediction, f)

    print(f"Prediction written to {result_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="hotpotqa", help="Dataset name (hotpotqa, musique, triviaqa)")
    # hotpot full
    parser.add_argument(
        "--data_path",
        type=str,
        default="../data/hotpotqa/hotpot_dev_distractor_v1.json",
        help="Path to the data file",
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

    # # pu-hotpot full
    # parser.add_argument('--data_path',type=str,default='../data/pu-hotpotqa/hotpot_dev_distractor_v1.json',help='Path to the data file')
    # parser.add_argument('--result_path',type=str,default='../output/pu-hotpot/pu-hotpot_dev_distractor_v1_full.json',help='Path to the result file')
    # parser.add_argument('--kg_dir',type=str,default='../data/pu-hotpotqa/kgs/extract_subkgs')
    # parser.add_argument('--persist_dir',type=str,default='../data/ollama_index/vector_pu_hotpotqa',help='Directory to store the index')

    parser.add_argument(
        "--model_name", type=str, default="llama3:8b", help="Ollama model name"
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
    args = parser.parse_args()
    main(args)
