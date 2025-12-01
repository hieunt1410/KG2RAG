import os
import string
import argparse
import pandas as pd
import ujson as json
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed


def ngram_overlap(span, sent, n=3):
    while (len(span) < n) or (len(sent) < n):
        n -= 1
    if n <= 0:
        return 0.0
    span = span.lower()
    sent = sent.lower()
    span_tokens = [token for token in span.split() if token not in string.punctuation]
    span_tokens = "".join(span_tokens)
    sent_tokens = [token for token in sent.split() if token not in string.punctuation]
    sent_tokens = "".join(sent_tokens)
    span_tokens = set([span_tokens[i : i + n] for i in range(len(span_tokens) - n + 1)])
    sent_tokens = set([sent_tokens[i : i + n] for i in range(len(sent_tokens) - n + 1)])
    overlap = span_tokens.intersection(sent_tokens)
    return float((len(overlap) + 0.01) / (len(span_tokens) + 0.01))


def extract_triplets(client, model_name, ctx, temperature, max_tokens):
    query = f'Extract triplets informative from the text following the examples. Make sure the triplet texts are only directly from the given text! Complete directly and strictly following the instructions without any additional words, line break nor space!\n{"-"*20}\nText: Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer.\nTriplets:<Scott Derrickson##born in##1966>$$<Scott Derrickson##nationality##America>$$<Scott Derrickson##occupation##director>$$<Scott Derrickson##occupation##screenwriter>$$<Scott Derrickson##occupation##producer>$$\n{"-"*20}\nText: A Kiss for Corliss is a 1949 American comedy film directed by Richard Wallace and written by Howard Dimsdale. It stars Shirley Temple in her final starring role as well as her final film appearance. Shirley Temple was named United States ambassador to Ghana and to Czechoslovakia and also served as Chief of Protocol of the United States.\nTriplets:<A Kiss for Corliss##cast member##Shirley Temple>$$<Shirley Temple##served as##Chief of Protocol>$$\n{"-"*20}\nText: {ctx}\nTriplets:'
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": query}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    resp = response.choices[0].message.content
    triplets = set()
    triplet_texts = resp.split("$$")
    for triplet_text in triplet_texts:
        if len(triplet_text) <= 6:
            continue
        triplet_text = triplet_text[1:-1]
        tokens = triplet_text.split("##")
        if not len(tokens) == 3:
            continue
        h = tokens[0].strip()
        r = tokens[1].strip()
        t = tokens[2].strip()
        if (
            ("no " in h)
            or ("no " in t)
            or ("unknown" in h)
            or ("unknown" in t)
            or ("No " in h)
            or ("No " in t)
            or ("Unknown" in h)
            or ("Unknown" in t)
            or ("null" in h)
            or ("null" in t)
            or ("Null" in h)
            or ("Null" in t)
            or ("NULL" in h)
            or ("NULL" in t)
            or ("NO" in h)
            or ("NO" in r)
            or ("NO" in t)
            or (h == t)
        ):
            continue
        if (r not in ctx) and (t not in ctx):
            continue

        triplets.add((h, r, t))
    triplets = [[h, r, t] for (h, r, t) in triplets]
    return triplets


def process_text(ent, text, seq, client, model_name, temperature, max_tokens):
    """Process a single text and extract triplets."""
    try:
        triplets = extract_triplets(client, model_name, text, temperature, max_tokens)
        if len(triplets) > 0:
            return (ent, seq, triplets)
    except Exception as e:
        print(f"\nError processing {ent} seq {seq}: {e}")
    return None


def extract_triplets_from_musique(
    data, client, model_name, temperature, max_tokens, workers
):
    ents = set()
    ent2text = dict()
    text2seq = dict()
    ent2triplets = dict()
    textcount = 0
    unique_textcount = 0

    # First pass: collect all unique texts and assign sequences
    texts_to_process = []

    for index, row in data.iterrows():
        question = row["question"]
        answer = row["answer"]
        ctxs = row["paragraphs"]
        current_ents = [ctx["title"] for ctx in ctxs]
        current_ents = sorted(current_ents, key=lambda x: len(x), reverse=True)

        for ctx in ctxs:
            ent = ctx["title"]
            text = ctx["paragraph_text"]
            ents.add(ent)

            if ent not in ent2text:
                ent2text[ent] = set()
            if text not in ent2text[ent]:
                ent2text[ent].add(text)
                unique_textcount += 1

            if ent not in text2seq:
                text2seq[ent] = dict()
            if text not in text2seq[ent]:
                text2seq[ent][text] = len(text2seq[ent])

            ctx["seq"] = text2seq[ent][text]
            textcount += 1

            # Add to processing queue
            texts_to_process.append((ent, text, ctx["seq"]))

    print(f"#ents: {len(ents)}")
    print(f"#total text: {textcount}")
    print(f"#unique text: {unique_textcount}")
    print(f"Using {workers} workers for concurrent processing")

    # Second pass: process texts concurrently
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit all tasks
        future_to_text = {
            executor.submit(
                process_text,
                ent,
                text,
                seq,
                client,
                model_name,
                temperature,
                max_tokens,
            ): (ent, seq)
            for ent, text, seq in texts_to_process
        }

        # Process completed tasks with progress bar
        with tqdm(total=len(future_to_text), desc="Extracting triplets") as pbar:
            for future in as_completed(future_to_text):
                try:
                    result = future.result()
                    if result is not None:
                        ent, seq, triplets = result
                        if ent not in ent2triplets:
                            ent2triplets[ent] = dict()
                        if seq not in ent2triplets[ent]:
                            ent2triplets[ent][seq] = list()
                        ent2triplets[ent][seq] = list(
                            set(ent2triplets[ent][seq]) | set(triplets)
                        )
                except Exception as e:
                    ent, seq = future_to_text[future]
                    print(f"\nError processing entity {ent} seq {seq}: {e}")
                finally:
                    pbar.update(1)

    return data, ent2triplets


def main(args):
    # Initialize OpenAI client
    print(f"Initializing OpenAI model: {args.model}")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    data_dir = "../../data/MuSiQue"
    data_path = os.path.join(data_dir, args.data_file)
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return

    print(f"Loading data from {data_path}")
    data = pd.read_json(data_path, lines=True)

    out_dir = "../../data/MuSiQue/kgs/extract_subkgs"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    mapped_data, ent2triplets = extract_triplets_from_musique(
        data, client, args.model, args.temperature, args.max_tokens, args.workers
    )

    kg_path = os.path.join(out_dir, "musique_kg.json")
    print(f"Saving extracted subkgs to {kg_path}")
    with open(kg_path, "w") as f:
        json.dump(ent2triplets, f)
    mapped_data_path = os.path.join(out_dir, "musique_ans_v1.0_dev_mapped.jsonl")
    print(f"Saving mapped data to {mapped_data_path}")
    mapped_data.to_json(mapped_data_path, orient="records", lines=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract knowledge graphs from MuSiQue using OpenAI"
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="musique_ans_v1.0_dev.jsonl",
        help="Data file name in MuSiQue directory",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model name (e.g., gpt-4o-mini, gpt-4o, gpt-3.5-turbo)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for OpenAI model (default: 0.0)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1000,
        help="Maximum tokens for OpenAI response (default: 1000)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent workers (default: 4)",
    )
    args = parser.parse_args()
    main(args)
