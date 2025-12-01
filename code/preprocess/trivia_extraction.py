import os
import argparse
import ujson as json
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock


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
    triplet_texts = resp.split('$$')
    for triplet_text in triplet_texts:
        if len(triplet_text) <= 6:
            continue
        triplet_text = triplet_text[1:-1]
        tokens = triplet_text.split('##')
        if not len(tokens) == 3:
            continue
        h = tokens[0].strip()
        r = tokens[1].strip()
        t = tokens[2].strip()
        if ('no ' in h) or ('no ' in t) or ('unknown' in h) or ('unknown' in t) or ('No ' in h) or ('No ' in t) or ('Unknown' in h) or ('Unknown' in t) or ('null' in h) or ('null' in t) or ('Null' in h) or ('Null' in t) or ('NULL' in h) or ('NULL' in t) or ('NO' in h) or ('NO' in r) or ('NO' in t) or (h==t):
            continue
        if (r not in ctx) and (t not in ctx):
            continue

        triplets.add((h, r, t))
    triplets = [[h,r,t] for (h,r,t) in triplets]
    return triplets


def process_entity(ent, ctx, client, model_name, temperature, max_tokens, out_dir, processed_ents, lock):
    """Process a single entity and extract triplets from all its contexts."""
    # Check if already processed (thread-safe)
    with lock:
        if ent in processed_ents:
            return None
        processed_ents.add(ent)

    # Check if file already exists
    out_path = os.path.join(out_dir, f'{ent.replace("/","_")}.json')
    if os.path.exists(out_path):
        return None

    # Extract triplets for each context
    entity_triplets = {}
    for i in range(len(ctx[1])):
        ctx_text = f'{ent}: {ctx[1][i]}'
        try:
            ext_triplets = extract_triplets(
                client, model_name, ctx_text, temperature, max_tokens
            )
            if len(ext_triplets) > 0:
                entity_triplets[i] = ext_triplets
        except Exception as e:
            print(f"\nError processing {ent} context {i}: {e}")
            continue

    # Save to file if we got any triplets
    if entity_triplets:
        with open(out_path, 'w') as f:
            json.dump(entity_triplets, f)
        return ent
    return None


def main(args):
    # Initialize OpenAI client
    print(f"Initializing OpenAI model: {args.model}")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    data_path = args.data_path
    print(f"Loading data from {data_path}")
    with open(data_path) as f:
        data = json.load(f)

    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Collect all entities to process
    entities_to_process = []
    processed_ents = set()
    lock = Lock()

    for sample in data:
        question = sample['question']
        answer = sample['answer']
        ctxs = sample['context']

        # Handle missing supporting_facts in TriviaQA
        if 'supporting_facts' in sample and sample['supporting_facts']:
            cands = [sp[0] for sp in sample['supporting_facts']]
        else:
            # If no supporting facts, process all entities
            cands = [ctx[0] for ctx in ctxs]

        for ctx in ctxs:
            ent = ctx[0]
            if ent not in processed_ents and ent in cands:
                entities_to_process.append((ent, ctx))
                processed_ents.add(ent)

    print(f"Found {len(entities_to_process)} unique entities to process")
    print(f"Using {args.workers} workers for concurrent processing")

    # Reset processed_ents for thread-safe tracking
    processed_ents = set()
    count = 0

    # Process entities concurrently
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        future_to_entity = {
            executor.submit(
                process_entity,
                ent,
                ctx,
                client,
                args.model,
                args.temperature,
                args.max_tokens,
                args.out_dir,
                processed_ents,
                lock,
            ): ent
            for ent, ctx in entities_to_process
        }

        # Process completed tasks with progress bar
        with tqdm(total=len(future_to_entity), desc="Extracting KGs") as pbar:
            for future in as_completed(future_to_entity):
                try:
                    result = future.result()
                    if result is not None:
                        count += 1
                except Exception as e:
                    ent = future_to_entity[future]
                    print(f"\nError processing entity {ent}: {e}")
                finally:
                    pbar.update(1)

    print(f'\n✅ Newly extracted entity KGs: {count}')
    print(f'📁 Output directory: {args.out_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract knowledge graphs from TriviaQA using OpenAI"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="../../data/trivia_qa/trivia.json",
        help="Path to the input data file",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="../../data/trivia_qa/kgs/extracted_subkgs",
        help="Output directory for extracted KGs",
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