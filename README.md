# Document Q&A CLI (RAG — LangChain v0.3)

আপনার নিজের PDF/TXT ফাইল থেকে প্রশ্নের উত্তর দেয় এমন একটি কমান্ড-লাইন অ্যাপ।
প্রতিটি উত্তরের সাথে কোন ফাইল (এবং PDF হলে কোন পৃষ্ঠা) থেকে এসেছে তা দেখায়।
ডকুমেন্টে উত্তর না থাকলে "জানি না" বলে — বানিয়ে কিছু বলে না।

## ১. সেটআপ (একবারই করতে হবে)

```bash
python -m venv venv
venv\Scripts\activate              # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
copy .env.example .env             # Mac/Linux: cp .env.example .env
```

তারপর `.env` ফাইল খুলে আপনার আসল key বসান:

```
OPENAI_API_KEY=sk-...
```

> ⚠️ `.env` ফাইল কখনো git-এ commit করবেন না (এটি `.gitignore`-এ আছে)।

## ২. ডকুমেন্ট রাখুন

`documents/` ফোল্ডারে আপনার `.pdf` বা `.txt` ফাইল রাখুন (সাব-ফোল্ডারও চলবে)।
উদাহরণ হিসেবে `documents/sample.txt` দেওয়া আছে।

## ৩. চালান

```bash
python main.py              # প্রশ্নোত্তর মোড; আগের ইনডেক্স থাকলে সেটাই ব্যবহার করে (খরচ নেই)
python main.py --rebuild    # নতুন ফাইল যোগ/বাদ দিলে আবার ইনডেক্স করুন
python main.py --chat       # কথোপকথন মোড — আগের প্রশ্ন মনে রাখে ("আরো বুঝিয়ে বলো")
python main.py --rebuild --chat
```

`Question:` এর পর প্রশ্ন লিখুন। বের হতে `exit`, `quit` বা `q` লিখুন।

প্রথমবার চালালে ডকুমেন্টগুলো embed হয় (OpenAI খরচ হয়) এবং `chroma_db/` ফোল্ডারে সেভ হয়।
পরের বার "Loaded existing vector store" দেখাবে — অর্থাৎ আবার embedding খরচ হয়নি।

## ৪. সেটিংস পরিবর্তন

সব সেটিংস `src/config.py` ফাইলে:

| সেটিং            | ডিফল্ট                    | মানে                                      |
| ---------------- | ------------------------ | ----------------------------------------- |
| `LLM_MODEL`      | `gpt-4o-mini`            | উত্তর দেওয়ার মডেল                         |
| `EMBEDDING_MODEL`| `text-embedding-3-small` | embedding মডেল                            |
| `CHUNK_SIZE`     | `1000`                   | প্রতিটি টুকরার দৈর্ঘ্য (অক্ষর)              |
| `CHUNK_OVERLAP`  | `200`                    | পাশাপাশি টুকরার মধ্যে মিল থাকা অংশ          |
| `TOP_K`          | `4`                      | কতগুলো টুকরা LLM-কে পাঠানো হবে              |
| `RETRIEVER_TYPE` | `simple`                 | নিচের টেবিল দেখুন                          |

| `RETRIEVER_TYPE` | কী করে                                         | প্রতি প্রশ্নে বাড়তি LLM কল |
| ---------------- | ---------------------------------------------- | ------------------------- |
| `simple`         | সাধারণ similarity search                       | ০                         |
| `hybrid`         | keyword (BM25) + semantic search একসাথে        | ০                         |
| `multi_query`    | প্রশ্নটি কয়েকভাবে লিখে খোঁজে                   | ১ (বাড়তি খরচ)             |
| `compression`    | পাওয়া টুকরা থেকে শুধু দরকারি অংশ রাখে           | ~`TOP_K` (বাড়তি খরচ)      |

`CHUNK_SIZE` বদলালে `python main.py --rebuild` চালান।

## ৫. ঐচ্ছিক: LangSmith মনিটরিং

`.env`-এ `LANGSMITH_API_KEY` দিলে প্রতিটি প্রশ্নের trace LangSmith-এ দেখা যাবে।
key না দিলে এটি নিজে থেকেই বন্ধ থাকে।

## ৬. টেস্ট (কোনো API খরচ নেই)

```bash
python -m pytest -q
```

## ৭. সাধারণ সমস্যা

| সমস্যা                                      | সমাধান                                              |
| ------------------------------------------ | --------------------------------------------------- |
| `OPENAI_API_KEY পাওয়া যায়নি`               | `.env.example` কপি করে `.env` বানান, key বসান        |
| `ModuleNotFoundError: src`                 | প্রজেক্ট ফোল্ডার থেকে `python main.py` চালান          |
| নতুন ফাইলের তথ্য উত্তরে আসছে না             | `python main.py --rebuild`                          |
| `RateLimitError` / `insufficient_quota`    | OpenAI অ্যাকাউন্টে credit যোগ করুন বা অপেক্ষা করুন     |
| উত্তর ভুল বা খালি                           | `TOP_K` বাড়ান, `CHUNK_SIZE` বদলান, বা `hybrid` চেষ্টা করুন |
