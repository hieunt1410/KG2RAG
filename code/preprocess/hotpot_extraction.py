import os
import asyncio
import ujson as json
from tqdm import tqdm
from openai import AsyncOpenAI


async def extract_triplets(client, ctx):
    query = f"Extract triplets informative from the text following the examples. Make sure the triplet texts are only directly from the given text! Complete directly and strictly following the instructions without any additional words, line break nor space!\n{'-' * 20}\nText: Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer.\nTriplets:<Scott Derrickson##born in##1966>$$<Scott Derrickson##nationality##America>$$<Scott Derrickson##occupation##director>$$<Scott Derrickson##occupation##screenwriter>$$<Scott Derrickson##occupation##producer>$$\n{'-' * 20}\nText: A Kiss for Corliss is a 1949 American comedy film directed by Richard Wallace and written by Howard Dimsdale. It stars Shirley Temple in her final starring role as well as her final film appearance. Shirley Temple was named United States ambassador to Ghana and to Czechoslovakia and also served as Chief of Protocol of the United States.\nTriplets:<A Kiss for Corliss##cast member##Shirley Temple>$$<Shirley Temple##served as##Chief of Protocol>$$\n{'-' * 20}\nText: {ctx}\nTriplets:"

    try:
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": query}],
            max_tokens=1000,
            temperature=0,
        )
        resp = resp.choices[0].message.content
    except Exception as e:
        print(f"Error processing text: {e}")
        return []

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


async def process_entity(client, ent, ctx, out_dir, semaphore):
    async with semaphore:
        out_path = os.path.join(out_dir, f"{ent.replace('/', '_')}.json")
        if os.path.exists(out_path):
            return 0

        # Process all contexts for this entity concurrently
        tasks = []
        for i in range(len(ctx[1])):
            if not i == 0:
                ctx_text = f"{ent}: {ctx[1][i]}"
            else:
                ctx_text = ctx[1][i]
            tasks.append(extract_triplets(client, ctx_text))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        entity_triplets = {}
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Error processing {ent} context {i}: {result}")
                continue
            ext_triplets = result
            if len(ext_triplets) == 0:
                continue
            entity_triplets[i] = ext_triplets

        # Save results if we have any triplets
        if entity_triplets:
            # Use async file I/O to avoid blocking
            def save_to_file():
                with open(out_path, "w") as f:
                    json.dump(entity_triplets, f)

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, save_to_file)
            return 1
        return 0


async def main():
    data_path = "../../data/hotpotqa/hotpotqa.json"
    with open(data_path) as f:
        data = json.load(f)

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    out_dir = "../../data/hotpotqa/kgs/extract_subkgs"
    os.makedirs(out_dir, exist_ok=True)

    # Collect all unique entities and their contexts
    entities_to_process = {}
    for sample in data:
        ctxs = sample["context"]
        for ctx in ctxs:
            ent = ctx[0]
            if ent not in entities_to_process:
                out_path = os.path.join(out_dir, f"{ent.replace('/', '_')}.json")
                if not os.path.exists(out_path):
                    entities_to_process[ent] = ctx

    print(f"Total entities to process: {len(entities_to_process)}")

    # Process entities concurrently with semaphore for rate limiting
    semaphore = asyncio.Semaphore(10)  # Limit to 10 concurrent requests

    tasks = [
        process_entity(client, ent, ctx, out_dir, semaphore)
        for ent, ctx in entities_to_process.items()
    ]

    count = 0
    with tqdm(total=len(tasks), desc="Processing entities") as pbar:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            count += result
            pbar.update(1)

    print(f"Newly extracted entity KGs number: {count}")


if __name__ == "__main__":
    asyncio.run(main())
