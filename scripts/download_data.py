from datasets import load_dataset
import json
import gdown
from bs4 import BeautifulSoup
from tqdm import tqdm
import unicodedata
import zipfile
import requests
import io
import os
import random
from sklearn.model_selection import train_test_split

import spacy
nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])

def main():
    # Category -> SIMPLE/EASY

    # --------
    # SQUAD2.0
    # --------
    corpus = []
    id = 1
    data = load_dataset("rajpurkar/squad_v2")

    for i in data["train"]:
        entry = {}

        entry["id"] = id
        entry["question"] = i["question"]
        try:
            entry["answer"] = i["answers"]["text"][0]
        except IndexError:
            continue

        doc = nlp(i["context"])
        entry["context"] = [sent.text.strip() for sent in doc.sents]

        entry["difficulty"] = "simple"

        corpus.append(entry)
        id += 1

    print(f"Length of SQUAD2.0 corpus is {len(corpus)}")

    with open("squad2.jsonl", "w", encoding="utf-8") as f:
        for entry in corpus:
            f.write(json.dumps(entry) + "\n")
    print("Finished saving SQUAD2.0 corpus")

    # -----------------
    # Natural Questions
    # -----------------
    def process_html_to_sentences(html_content):
        soup = BeautifulSoup(html_content, "html.parser")

        clean_text = soup.get_text(separator="\n")
        normalized_text = " ".join(clean_text.split())

        doc = nlp(normalized_text)

        return [sent.text.strip() for sent in doc.sents]

    id = 86822
    corpus = []

    file_id = "1hbMLcIparxKIUQ9DEIILIpgAQr9kXWY6"
    output = "nqa_data.jsonl"

    # Note that the file is large, so it may take some time to download
    gdown.download(id=file_id, output=output, quiet=False)

    nqa = []

    with open(output, "r", encoding="utf-8") as f:
        for line in f:
            try:
                # We only want to process the first 80,000 entries to keep the dataset manageable
                if len(nqa) == 80_000:
                    break
                json_object = json.loads(line)
                nqa.append(json_object)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line: {line.strip()} - {e}")

    for i in nqa:
        entry = {}

        entry["id"] = id
        entry["question"] = i["question_text"]
        if len(i["annotations"][0]["short_answers"]) == 0:
            continue
        split_text = i["document_text"].split(" ")
        start = i["annotations"][0]["short_answers"][0]["start_token"]
        end = i["annotations"][0]["short_answers"][0]["end_token"]
        ans = " ".join(split_text[start:end])

        if ans.strip() == "":
            continue

        entry["answer"] = ans

        start = i["annotations"][0]["long_answer"]["start_token"]
        end = i["annotations"][0]["long_answer"]["end_token"]
        context = " ".join(split_text[start:end]).strip()
        entry["context"] = process_html_to_sentences(context)

        entry["difficulty"] = "simple"

        corpus.append(entry)
        id += 1

    print(f"Length of NQA corpus is {len(corpus)}")

    def redistribute_sentences(sentences):
        n = len(sentences)
        if n == 0:
            return ["", "", ""]

        base_size = n // 3
        remainder = n % 3

        # Determine how many sentences go into each of the 3 chunks
        # Case 0: [base, base, base] Case 1: [base+1, base, base]
        # Case 2: [base+1, base+1, base]
        counts = [base_size + (1 if i < remainder else 0) for i in range(3)]

        chunks = []
        start = 0
        for count in counts:
            chunk_text = " ".join(sentences[start : start + count])
            chunks.append(chunk_text)
            start += count

        return chunks

    with open("nqa.jsonl", "w", encoding="utf-8") as f:
        for entry in corpus:
            entry["context"] = redistribute_sentences(entry["context"])
            f.write(json.dumps(entry) + "\n")
    print("Finished saving NQA corpus")

    # --------
    # TriviaQA
    # --------

    id = 114751
    corpus = []

    triviaqa = load_dataset("mandarjoshi/trivia_qa", "rc")
    triviaqa = triviaqa["train"]

    for i in triviaqa:
        entry = {}

        entry["id"] = id
        entry["question"] = i["question"]
        ans = i["answer"]["value"]

        if ans.strip() == "":
            continue

        entry["answer"] = ans

        context = []

        for c in range(len(i["search_results"]["title"])):
            text = "\n".join(list(dict.fromkeys(i["search_results"]["search_context"][c].encode("utf-8").decode("utf-8").split("\n"))))

            if ans.lower() in text.lower():
                context.append(text)

        if len(context) == 0:
            continue

        entry["context"] = context

        entry["difficulty"] = "simple"

        corpus.append(entry)
        id += 1

    print(f"Length of TriviaQA corpus is {len(corpus)}")

    corpus = [item for item in corpus if len(item["context"]) <= 7]
    print("Length after : ", len(corpus))

    with open("triviaqa.jsonl", "w", encoding="utf-8") as f:
        for entry in corpus:
            f.write(json.dumps(entry) + "\n")
    print("Finished saving TriviaQA corpus")


    files_to_merge = ["squad2.jsonl", "nqa.jsonl", "triviaqa.jsonl"]
    output_file = "simple_data.jsonl"

    def normalize_context(sentences):
        """Applies NFKC normalization to a list of strings."""
        if not sentences:
            return []
        # Normalize each sentence in the list
        return [unicodedata.normalize("NFKC", s) for s in sentences]

    with open(output_file, "w", encoding="utf-8") as outfile:
        for filename in files_to_merge:
            with open(filename, "r", encoding="utf-8") as infile:
                for line in tqdm(infile, desc=f"Merging {filename}"):
                    if line.strip():
                        # Normalize context in each line before writing
                        entry = json.loads(line.strip())
                        entry["context"] = normalize_context(entry["context"])
                        outfile.write(json.dumps(entry) + "\n")
    print("Finished merging.")


    # Category -> MEDIUM

    # --------
    # HotpotQA
    # --------
    id = 183330
    corpus = []

    hotpotqa = load_dataset("hotpotqa/hotpot_qa", "fullwiki")
    hotpotqa = hotpotqa["train"]

    for i in hotpotqa:
        entry = {}

        entry["id"] = id
        entry["difficulty"] = "medium"

        entry["question"] = i["question"]
        ans = i["answer"].strip()

        if ans == "":
            continue

        entry["answer"] = ans

        docs = []

        for d in range(len(i["context"]["title"])):
            text = " ".join(i["context"]["sentences"][d])
            docs.append(text)

        if docs == []:
            continue

        entry["context"] = docs

        corpus.append(entry)
        id += 1

    print(f"Length of HotpotQA corpus is {len(corpus)}")

    with open("hotpotqa.jsonl", "w", encoding="utf-8") as f:
        for entry in corpus:
            f.write(json.dumps(entry) + "\n")
    print("Finished saving HotpotQA corpus")

    # -------
    # MuSiQue
    # -------
    id = 273777
    corpus = []

    file_id = "1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h"
    output = "musique.zip"

    gdown.download(id=file_id, output=output, quiet=False)

    with zipfile.ZipFile("musique.zip", 'r') as zip_ref:
        zip_ref.extractall("./musique")

    in_dir = "./musique/data/musique_full_v1.0_train.jsonl"

    musique = []
    with open(in_dir, "r", encoding="utf-8") as f:
        for line in f:
            try:
                json_object = json.loads(line.strip())
                musique.append(json_object)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line: {line.strip()}. Error: {e}")

    for i in musique:
        if not i["answerable"]:
            continue

        entry = {}

        entry["id"] = id
        entry["difficulty"] = "medium"

        entry["question"] = i["question"]
        ans = i["answer"]

        if ans.strip() == "":
            continue

        entry["answer"] = ans

        docs = []

        for d in i["paragraphs"]:
            docs.append(str(d["paragraph_text"]))

        entry["context"] = docs

        corpus.append(entry)
        id += 1

    print(f"Length of MuSiQue corpus is {len(corpus)}")

    with open("musique.jsonl", "w", encoding="utf-8") as f:
        for entry in corpus:
            f.write(json.dumps(entry) + "\n")
    print("Finished saving MuSiQue corpus")

    # -----------
    # Multihop QA
    # -----------
    id = 293715
    corpus = []

    def download_github_raw_file(url, save_path):
        """
        Downloads a file from a raw GitHub URL to a specified local path.
        """
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Successfully downloaded file to {save_path}")

        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")

    github_url = 'https://huggingface.co/datasets/alabnii/morehopqa/raw/main/data/with_human_verification.json'
    local_file_path = 'with_human_verification.json'

    download_github_raw_file(github_url, local_file_path)

    with open("with_human_verification.json", "r") as f:
        multihop = json.load(f)

    for i in multihop:
        entry = {}

        entry["id"] = id
        entry["difficulty"] = "medium"

        entry["question"] = i["question"]
        ans = i["answer"]

        if ans.strip() == "":
            continue

        entry["answer"] = ans

        docs = []

        for d in i["context"]:
            content = " ".join(d[1])
            docs.append(str(content))

        entry["context"] = docs

        corpus.append(entry)
        id += 1

    print(f"Length of Multihop QA corpus is {len(corpus)}")

    with open("multihopqa.jsonl", "w", encoding="utf-8") as f:
        for entry in corpus:
            f.write(json.dumps(entry) + "\n")
    print("Finished saving Multihop QA corpus")

    files_to_merge = ["hotpotqa.jsonl", "musique.jsonl", "multihopqa.jsonl"]
    output_file = "medium_data.jsonl"

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for file_name in files_to_merge:
            with open(file_name, 'r', encoding='utf-8') as infile:
                for line in tqdm(infile, desc=f"Merging {file_name}"):
                    if line.strip():
                        outfile.write(line)
    print(f"Successfully merged into {output_file}")


    # Category -> HARD

    # -----------
    # Strategy QA
    # -----------
    id = 294833
    corpus = []

    train = "https://raw.githubusercontent.com/eladsegal/strategyqa/refs/heads/main/data/strategyqa/train.json"
    paragraphs = "https://raw.githubusercontent.com/eladsegal/strategyqa/refs/heads/main/data/strategyqa/strategyqa_train_paragraphs.json"

    response = requests.get(train)
    if response.status_code == 200:
        strategyqa = response.json()
        print("Success! Data loaded.")
    else:
        print(f"Failed to download. Status code: {response.status_code}")

    response = requests.get(paragraphs)
    if response.status_code == 200:
        strategyqa_para = response.json()
        print("Success! Paragraphs data loaded.")
    else:
        print(f"Failed to download paragraphs. Status code: {response.status_code}")

    def extract_evidence_keys(data, exclude_set):
        for item in data:
            if isinstance(item, list):
                # Recursively explore nested lists
                yield from extract_evidence_keys(item, exclude_set)
            else:
                if item not in exclude_set:
                    yield item

    for i in strategyqa:
        entry = {}

        entry["id"] = id
        entry["difficulty"] = "hard"

        entry["question"] = i["question"]
        ans = i["answer"]

        entry["answer"] = ans

        docs = []

        exclude = {'no_evidence', 'operation'}
        evidences = set()

        for d in i["evidence"]:
            results = list(extract_evidence_keys(d, exclude))
            for results in results:
                evidences.add(results)

        for ev in evidences:
            content = strategyqa_para[ev]["content"]
            docs.append(content)

        entry["context"] = docs

        corpus.append(entry)
        id += 1

    print(f"Length of Strategy QA corpus is {len(corpus)}")

    with open("strategyqa.jsonl", "w", encoding="utf-8") as f:
        for entry in corpus:
            f.write(json.dumps(entry) + "\n")
    print("Finished saving Strategy QA corpus")

    # -------------
    # Climate FEVER
    # -------------
    id = 297123
    corpus = []

    climate_fever = load_dataset("tdiggelm/climate_fever")
    climate_fever = climate_fever["test"]

    for i in climate_fever:
        entry = {}

        entry["id"] = id
        entry["difficulty"] = "hard"
        entry["question"] = "Is the following claim true or false: " + i["claim"]

        ans = i["claim_label"]

        if str(ans).strip() == "":
            continue

        if ans == 0:
            ans = "True"
        elif ans == 1:
            ans = "False"
        else:
            continue

        entry["answer"] = ans

        docs = []

        ev = i["evidences"]

        for j in ev:
            evid = j["evidence"]
            docs.append(evid)

        entry["context"] = docs

        corpus.append(entry)
        id += 1

    print(f"Length of Climate FEVER corpus is {len(corpus)}")

    with open("climate_fever.jsonl", "w", encoding="utf-8") as f:
        for entry in corpus:
            f.write(json.dumps(entry) + "\n")
    print("Finished saving Climate FEVER corpus")

    # -----
    # FEVER
    # -----
    id = 298030
    corpus = []

    def extract_wiki_id_line_num(nested_list):
        return [(item[2], item[3]) for sublist in nested_list for item in sublist]

    url = "https://fever.ai/download/fever/train.jsonl"
    response = requests.get(url)
    if response.status_code == 200:
        fever = [json.loads(line) for line in response.text.strip().split("\n")]

    verifiable_data = []
    for i in fever:
        if i["verifiable"] == "VERIFIABLE" and i["label"] != "NOT ENOUGH INFO":
            verifiable_data.append(i)

    print("Number of VERIFIABLE entries =", len(verifiable_data))

    url = "https://fever.ai/download/fever/wiki-pages.zip"
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extractall("./fever")
        print(f"Success! Files extracted to the './fever' folder.")
    else:
        print(f"Failed to download. Status code: {response.status_code}")

    wiki_path = "./fever/wiki-pages/"
    file_names = os.listdir(wiki_path)
    file_names = sorted(file_names)

    pages = []

    for f in file_names:
        p = wiki_path + f
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                pages.append(json.loads(line))

    for i in verifiable_data:
        if len(corpus) % 1000 == 0:
            print(f"Length of corpus so far while adding FEVER is {len(corpus)}")
        if len(corpus) > 20000: # Limiting to first 20,000 entries
            break

        entry = {}

        entry["id"] = id
        entry["difficulty"] = "hard"

        entry["question"] = "Is the following claim true or false: " + i["claim"]

        ans = i["label"]

        if ans.strip() == "":
            continue

        if ans == "SUPPORTS":
            ans = "True"
        elif ans == "REFUTES":
            ans = "False"
        else:
            continue

        ans = ans.lower()

        entry["answer"] = ans

        docs = []

        ev = i["evidence"]
        ids_num = extract_wiki_id_line_num(ev)

        for i in ids_num:
            for p in pages:
                if p["id"] == i[0]:
                    content = p["text"]
                    docs.append(content)

        docs = list(set(tuple(docs)))
        entry["context"] = docs

        corpus.append(entry)
        id += 1

    print(f"Length of FEVER corpus is {len(corpus)}")

    with open("fever.jsonl", "w", encoding="utf-8") as f:
        for entry in corpus:
            f.write(json.dumps(entry) + "\n")
    print("Finished saving FEVER corpus")

    files_to_merge = ["strategyqa.jsonl", "climate_fever.jsonl", "fever.jsonl"]
    output_file = "hard_data.jsonl"

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for file_name in files_to_merge:
            with open(file_name, 'r', encoding='utf-8') as infile:
                for line in tqdm(infile, desc=f"Merging {file_name}"):
                    if line.strip():
                        outfile.write(line)
    print(f"Successfully merged into {output_file}")


    # TRAIN/VAL/TEST SPLIT
    simple = "./simple_data.jsonl"
    medium = "./medium_data.jsonl"
    hard = "./hard_data.jsonl"

    def train_val_test_split(input_file, label="simple", train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        with open(input_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]

        train_data, temp_data = train_test_split(data, test_size=(1 - train_ratio), random_state=42)

        # Calculate the relative proportion of the test set
        relative_test_size = test_ratio / (val_ratio + test_ratio)
        val_data, test_data = train_test_split(temp_data, test_size=relative_test_size, random_state=42)

        def save_jsonl(data, filename):
            with open(filename, 'w', encoding='utf-8') as f:
                for entry in data:
                    f.write(json.dumps(entry) + '\n')

        save_jsonl(train_data, label+'_train.jsonl')
        save_jsonl(val_data, label+'_val.jsonl')
        save_jsonl(test_data, label+'_test.jsonl')

        print("Completed : ", label)

    train_val_test_split(input_file=simple, label="simple")
    train_val_test_split(input_file=medium, label="medium")
    train_val_test_split(input_file=hard, label="hard")

    def combine_and_shuffle(input_files, output_file, seed=42):
        combined_data = []

        for file_path in input_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip(): # Avoid empty lines
                        combined_data.append(json.loads(line))

        # Setting a seed
        random.seed(seed)
        random.shuffle(combined_data)

        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in combined_data:
                f.write(json.dumps(entry) + '\n')

        print(f"Successfully combined {len(combined_data)} records into {output_file}")

    trains = ["./simple_train.jsonl", "./medium_train.jsonl", "./hard_train.jsonl"]
    vals = ["./simple_val.jsonl", "./medium_val.jsonl", "./hard_val.jsonl"]
    tests = ["./simple_test.jsonl", "./medium_test.jsonl", "./hard_test.jsonl"]

    combine_and_shuffle(trains, "train.jsonl")
    combine_and_shuffle(vals, "val.jsonl")
    combine_and_shuffle(tests, "test.jsonl")

    # Trim : Train 150000, Val 27000, Test 23000
    def trim(input_file, output_file, num):
        with open(input_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]

        if num <= len(data):
            trimmed_data = data[:num]
        else:
            print("Trim value should be <= length of file")
            return

        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in trimmed_data:
                f.write(json.dumps(entry) + '\n')

        print(f"Successfully trimmed {len(trimmed_data)} records into {output_file}")

    trim("./train.jsonl","./train_trim.jsonl", 150000)
    trim("./val.jsonl","./val_trim.jsonl", 27000)
    trim("./test.jsonl","./test_trim.jsonl", 23000)

    print("All done!")


if __name__ == "__main__":
    main()