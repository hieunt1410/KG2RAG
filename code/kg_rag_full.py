import argparse
import json
import os
from math import ceil
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def process_question(sample, retriever, postprocessors):
    """Process a single question and return results"""
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
        pass

    # Extract supporting facts from retrieved nodes
    sps = [
        [
            node.id_.split("##")[0],
            int(node.id_.split("##")[1]),
        ]
        for node in nodes
    ]

    # Return results for this sample
    return sample_id, sps


def kg_rag_parallel(
    questions,
    corpora,
    doc2kg,
    top_k=5,
    persist_dir=None,
    reranker="../model/bge-reranker-large",
    dataset="hotpotqa",
    embedding_batch_size=1000,
    question_batch_size=50,
    num_workers=1,
):
    prediction = {"answer": {}, "sp": {}}

    # Process corpora in batches
    chunks_index = dict()
    ents = set()

    # First pass: collect all entities and build chunks_index structure
    print("Collecting entities and chunk indices...")
    for sample in corpora:
        for ctx in sample["context"]:
            ent = ctx[0]
            ents.add(ent)
            if ent not in chunks_index:
                chunks_index[ent] = dict()
            for i in range(len(ctx[1])):
                if str(i) not in chunks_index[ent]:
                    chunks_index[ent][str(i)] = f"{ent}: {ctx[1][i]}"

    # Process embeddings in batches
    if (persist_dir is not None) and os.path.exists(persist_dir):
        print("Load index from persist dir")
        sc = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(sc)
    else:
        print("Creating index with batched embeddings...")

        # Create TextNode objects with progress tracking
        all_chunks = []
        print("Collecting text chunks...")
        for ent, chunks in tqdm(chunks_index.items(), desc="Processing entities"):
            for chunk_id, text in chunks.items():
                all_chunks.append(TextNode(text=text, id_=f"{ent}##{chunk_id}"))

        print(f"\nTotal chunks to embed: {len(all_chunks)}")
        print(f"Batch size: {embedding_batch_size}")
        print(f"Total batches: {ceil(len(all_chunks) / embedding_batch_size)}")

        # Process in batches with progress bar
        print("\nCreating embeddings in batches...")

        # If we have a lot of chunks, process in batches to show progress
        if len(all_chunks) > embedding_batch_size * 2:  # Only use custom batching for large datasets
            index = None
            for i in tqdm(range(0, len(all_chunks), embedding_batch_size),
                         desc=f"Embedding batches ({embedding_batch_size} chunks/batch)",
                         total=ceil(len(all_chunks) / embedding_batch_size)):
                batch = all_chunks[i:i + embedding_batch_size]

                if index is None:
                    index = VectorStoreIndex(batch, show_progress=False)
                else:
                    for node in batch:
                        index.insert(node)
        else:
            # For smaller datasets, let VectorStoreIndex handle its own progress
            index = VectorStoreIndex(all_chunks, show_progress=True)

        if persist_dir is not None:
            print("\nSaving index to disk...")
            os.makedirs(persist_dir, exist_ok=True)
            index.storage_context.persist(persist_dir=persist_dir)
            print(f"Index saved to {persist_dir}")

    print(f"Index ready in persist dir {persist_dir}")
    retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)
    # Commented out answering parts - focusing on retrieval only
    # qa_rag_template_str = "Context information is below.\n{context_str}\nGive a short factoid answer (as few words as possible).\nQ: Were Scott Derrickson and Ed Wood of the same nationality?\nA: Yes.\nQ: Who was born earlier, Emma Bull or Virginia Woolf?\nA: Adeline Virginia Woolf.\nQ: The arena where the Lewiston Maineiacs played their home games can seat how many people?\nA: 3,677 seated.\nQ: What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?\nA: Chief of Protocol.\n---------------------\nQ: {query_str}\nA: "
    # qa_rag_prompt_template = PromptTemplate(qa_rag_template_str)
    # response_synthesizer = get_response_synthesizer(
    #     response_mode=ResponseMode.COMPACT, text_qa_template=qa_rag_prompt_template
    # )

    kg_post_processor1 = KGRetrievePostProcessor(
        dataset=dataset, ents=ents, doc2kg=doc2kg, chunks_index=chunks_index
    )
    kg_post_processor2 = GraphFilterPostProcessor(
        dataset=dataset,
        topk=top_k,
        use_tpt=False,  # Default value
        ents=ents,
        doc2kg=doc2kg,
        chunks_index=chunks_index,
        reranker_model=reranker,  # Pass model name, thread-local instances created on demand
    )

    # Create a retriever with postprocessors for retrieval only
    # We'll directly use the retriever with postprocessors to get nodes
    postprocessors = [
        kg_post_processor1,
        kg_post_processor2,
        NaivePostprocessor(dataset=dataset),
    ]

    # Process questions in batches to manage memory
    print(f"Processing {len(questions)} questions in batches of {question_batch_size}...")
    print(f"Using {num_workers} workers for concurrent processing")

    all_results = []
    sps_count = []

    # Process questions in batches
    for batch_start in tqdm(range(0, len(questions), question_batch_size),
                           desc="Question batches"):
        batch_end = min(batch_start + question_batch_size, len(questions))
        batch_questions = questions[batch_start:batch_end]

        # Process each question in the batch concurrently
        batch_results = []
        if num_workers > 1:
            # Use concurrent processing
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Submit all tasks
                future_to_sample = {
                    executor.submit(process_question, sample, retriever, postprocessors): sample
                    for sample in batch_questions
                }

                # Collect results as they complete
                for future in as_completed(future_to_sample):
                    sample = future_to_sample[future]
                    try:
                        result = future.result()
                        batch_results.append(result)
                        sps_count.append(len(result[1]))  # Track SPs count
                    except Exception as e:
                        print(f"Error processing question {sample['_id']}: {e}")
                        batch_results.append((sample["_id"], []))
                        sps_count.append(0)
        else:
            # Sequential processing (original behavior)
            for sample in tqdm(batch_questions, desc="Processing batch", leave=False):
                try:
                    result = process_question(sample, retriever, postprocessors)
                    batch_results.append(result)
                    sps_count.append(len(result[1]))  # Track SPs count
                except Exception as e:
                    print(f"Error processing question {sample['_id']}: {e}")
                    batch_results.append((sample["_id"], []))
                    sps_count.append(0)

        all_results.extend(batch_results)

        # Force garbage collection to free memory
        import gc
        gc.collect()

    # Process results and fill prediction dictionary
    for sample_id, sps in all_results:
        prediction["answer"][sample_id] = ""  # Empty answer since we're only doing retrieval
        prediction["sp"][sample_id] = sps

    if sps_count:
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
    loaded_count = 0
    for ent in tqdm(ents, desc="Loading knowledge graphs"):
        subkg_path = os.path.join(kg_dir, f"{ent.replace('/', '_')}.json")
        if not os.path.exists(subkg_path):
            continue
        try:
            with open(subkg_path, "r", encoding="utf-8") as fin:
                subkg = json.load(fin)
                if subkg and len(subkg.keys()) > 0:
                    for seq in list(subkg.keys()):
                        for i, triplet in enumerate(subkg[seq]):
                            h, r, t = triplet
                            if (ngram_overlap(h, ent) >= 0.90) or (
                                ngram_overlap(ent, h) >= 0.90
                            ):
                                h = ent
                            if (ngram_overlap(t, ent) >= 0.90) or (
                                ngram_overlap(ent, t) >= 0.90
                            ):
                                t = ent
                            subkg[seq][i] = (h, r, t)
                        if len(subkg[seq]) == 0:
                            del subkg[seq]
                    if len(subkg.keys()) > 0:
                        doc2kg[ent] = subkg
                        loaded_count += 1
        except Exception as e:
            print(f"Error loading KG for {ent}: {e}")
            continue

    print(f"\nLoaded {loaded_count} knowledge graphs out of {len(ents)} entities")

    # Commented out LLM initialization since we're only doing retrieval
    # model_name = args.model_name
    # print('Init Ollama model')
    # Settings.llm = Ollama(model=model_name,request_timeout=200)
    embed_model_name = args.embed_model_name
    print("Init Ollama embedding")
    Settings.embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
    top_k = args.top_k
    embedding_batch_size = args.embedding_batch_size
    question_batch_size = args.question_batch_size
    persist_dir = args.persist_dir
    reranker = args.reranker
    prediction = kg_rag_parallel(
        questions,
        corpora,
        doc2kg,
        top_k=top_k,
        embedding_batch_size=embedding_batch_size,
        question_batch_size=question_batch_size,
        persist_dir=persist_dir,
        dataset=args.dataset,
        reranker=reranker,
        num_workers=args.num_workers,
    )

    result_path = args.result_path
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(prediction, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="hotpotqa", help="Dataset name")
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
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model name",
    )
    parser.add_argument(
        "--reranker",
        type=str,
        default="BAAI/bge-reranker-large",
        help="Reranker model name",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of top-k retrieved documents",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of workers for parallel processing",
    )
    parser.add_argument(
        "--embedding_batch_size",
        type=int,
        default=1000,
        help="Batch size for processing embeddings",
    )
    parser.add_argument(
        "--question_batch_size",
        type=int,
        default=200,
        help="Batch size for processing questions (to manage memory)",
    )
    args = parser.parse_args()
    main(args)
