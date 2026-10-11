import json, os, time, html
from contextlib import nullcontext
from typing import List, Literal

import torch
import streamlit as st
from pydantic import BaseModel
from json_repair import repair_json
from langchain_core.output_parsers import PydanticOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForCausalLM
from pypdf import PdfReader

st.set_page_config(page_title="Interview Question Generator", page_icon="🎯", layout="wide")

BASE = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER = "qlora_adapter"          # بيتعمل في خطوة الـ Fine-tuning

SYSTEM = ("You are an expert technical interviewer. Given a candidate CV, a job title and reference "
          "material, return ONLY valid JSON, no extra text. Keys: candidate_level (exactly one of: Junior, Mid, Senior) "
          "and questions (exactly 5 items). Each item has: question (must mention a specific project or skill from the CV; "
          "never copy reference questions word for word), skill_tested, difficulty (exactly one of: Easy, Medium, Hard), "
          "ideal_answer.")

# ===== HELPERS START =====
class InterviewQuestion(BaseModel):
    question: str
    skill_tested: str
    difficulty: Literal["Easy", "Medium", "Hard"]
    ideal_answer: str


class InterviewQuestions(BaseModel):
    candidate_level: Literal["Junior", "Mid", "Senior"]
    questions: List[InterviewQuestion]


parser = PydanticOutputParser(pydantic_object=InterviewQuestions)

LEVELS = {"junior": "Junior", "entry": "Junior", "beginner": "Junior", "مبتدئ": "Junior", "جونيور": "Junior",
          "mid": "Mid", "mid-level": "Mid", "middle": "Mid", "intermediate": "Mid", "متوسط": "Mid",
          "senior": "Senior", "expert": "Senior", "advanced": "Senior", "متقدم": "Senior",
          "خبير": "Senior", "سينيور": "Senior"}
DIFFS = {"easy": "Easy", "basic": "Easy", "سهل": "Easy",
         "medium": "Medium", "moderate": "Medium", "intermediate": "Medium", "متوسط": "Medium",
         "hard": "Hard", "difficult": "Hard", "advanced": "Hard", "صعب": "Hard"}


def parse_output(raw):
    """Model text -> (validated InterviewQuestions, info). Raises on failure."""
    s = raw.find("{")
    if s == -1:
        raise ValueError("No JSON object found in model output")
    e = raw.rfind("}")
    repaired = False
    try:
        obj = json.loads(raw[s:e + 1])
    except Exception:
        repaired = True
        obj = repair_json(raw[s:], return_objects=True)
    if not isinstance(obj, dict):
        raise ValueError("No JSON object found in model output")

    normalized = False
    lvl_raw = str(obj.get("candidate_level", "")).strip()
    lvl = LEVELS.get(lvl_raw.lower())
    if lvl:
        normalized = normalized or lvl != lvl_raw
        obj["candidate_level"] = lvl
    qs = obj.get("questions")
    qs = qs if isinstance(qs, list) else []
    need = {"question", "skill_tested", "difficulty", "ideal_answer"}
    kept = [q for q in qs if isinstance(q, dict) and need <= q.keys()]   # نشيل السؤال الناقص/المقطوع
    if len(kept) != len(qs):
        repaired = True
    obj["questions"] = qs = kept
    for q in qs:
        if isinstance(q, dict):
            d_raw = str(q.get("difficulty", "")).strip()
            d = DIFFS.get(d_raw.lower())
            if d:
                normalized = normalized or d != d_raw
                q["difficulty"] = d

    result = parser.parse(json.dumps(obj, ensure_ascii=False))
    return result, {"repaired": repaired, "normalized": normalized}


TXT = {
    "ar": {
        "title": "مولّد أسئلة المقابلات الذكي",
        "sub": "ارفع السيرة الذاتية واكتب الوظيفة المطلوبة، وستحصل على أسئلة مقابلة مخصصة مع الإجابات النموذجية.",
        "settings": "⚙️ الإعدادات",
        "rag": "استخدام RAG (مراجع من قاعدة المعرفة)",
        "ft": "استخدام الموديل المدرَّب (Fine-tuned)",
        "ft_missing": "الموديل المدرَّب غير متاح بعد. يحتاج تنفيذ خطوة التدريب أولًا.",
        "job": "المسمى الوظيفي",
        "upload": "ارفع السيرة الذاتية (PDF)",
        "paste": "أو الصق نص السيرة الذاتية",
        "sample": "جرّب سيرة ذاتية تجريبية",
        "go": "ولّد الأسئلة",
        "empty": "ارفع ملف PDF أو الصق نص السيرة الذاتية أولًا.",
        "spinner": "جاري توليد الأسئلة...",
        "loading": "جاري تحميل الموديل (أول مرة تستغرق حوالي دقيقة)...",
        "hint": "ستظهر النتائج هنا بعد الضغط على «ولّد الأسئلة».",
        "level": "المستوى المقدَّر",
        "skill": "المهارة",
        "answer": "الإجابة النموذجية",
        "refs": "المراجع المسترجعة (RAG)",
        "dl": "تحميل النتيجة كملف نصي",
        "fail": "تعذّر تحويل مخرجات الموديل إلى شكل صحيح. جرّب مرة أخرى.",
        "raw": "المخرجات الخام",
        "q": "س",
        "valid": "المخرجات كانت بصيغة صحيحة من أول مرة",
        "fixed": "المخرجات احتاجت إصلاحًا تلقائيًا",
        "time": "الزمن",
        "sec": "ثانية",
        "levels": {"Junior": "مبتدئ", "Mid": "متوسط", "Senior": "متقدم"},
        "diffs": {"Easy": "سهل", "Medium": "متوسط", "Hard": "صعب"},
    },
    "en": {
        "title": "AI Interview Question Generator",
        "sub": "Upload a CV and enter the target job to get tailored interview questions with model answers.",
        "settings": "⚙️ Settings",
        "rag": "Use RAG (references from the knowledge base)",
        "ft": "Use fine-tuned model",
        "ft_missing": "Fine-tuned model not available yet. Run the training step first.",
        "job": "Job title",
        "upload": "Upload CV (PDF)",
        "paste": "or paste the CV text",
        "sample": "Try a sample CV",
        "go": "Generate questions",
        "empty": "Upload a PDF or paste the CV text first.",
        "spinner": "Generating questions...",
        "loading": "Loading the model (first run takes about a minute)...",
        "hint": "Results will appear here after you click “Generate questions”.",
        "level": "Estimated level",
        "skill": "Skill",
        "answer": "Model answer",
        "refs": "Retrieved references (RAG)",
        "dl": "Download as text file",
        "fail": "Could not convert the model output to a valid format. Please try again.",
        "raw": "Raw output",
        "q": "Q",
        "valid": "Output was valid on the first try",
        "fixed": "Output needed automatic repair",
        "time": "Time",
        "sec": "s",
        "levels": {"Junior": "Junior", "Mid": "Mid", "Senior": "Senior"},
        "diffs": {"Easy": "Easy", "Medium": "Medium", "Hard": "Hard"},
    },
}


def to_text(r, lang, job):
    t = TXT[lang]
    lines = [t["title"], f"{t['job']}: {job}", f"{t['level']}: {t['levels'][r.candidate_level]}", "=" * 40, ""]
    for i, q in enumerate(r.questions, 1):
        lines += [f"{t['q']}{i}) [{t['diffs'][q.difficulty]}] {q.question}",
                  f"   {t['skill']}: {q.skill_tested}",
                  f"   {t['answer']}: {q.ideal_answer}",
                  ""]
    return "\n".join(lines)


def rtl(text, cls="", lang="ar"):
    d = "rtl" if lang == "ar" else "ltr"
    return f'<div dir="{d}" class="{cls}">{html.escape(text)}</div>'
# ===== HELPERS END =====

SAMPLE_CV = ("Skills: Python, Django, PostgreSQL, Docker. Experience: 3 years backend developer at a fintech startup. "
             "Project: e-commerce platform with 50k products and 2k daily orders; built REST APIs with JWT auth and "
             "reduced page load time by 40% using caching.")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap');
.hero, .qtext, .atext, .stMarkdown p, label p, textarea, input { font-family: 'Cairo', sans-serif; }
.hero {background: linear-gradient(135deg,#4f46e5,#7c3aed 60%,#06b6d4); padding: 26px 32px;
       border-radius: 18px; color: #fff; margin-bottom: 18px;}
.hero h1 {margin: 0; font-size: 1.9rem; color: #fff; padding: 0;}
.hero p {margin: 6px 0 0; opacity: .93; font-size: 1.02rem;}
.chip {display: inline-block; padding: 3px 12px; border-radius: 999px; font-size: .82rem;
       font-weight: 600; margin: 0 0 0 6px;}
.c-easy {background: #dcfce7; color: #166534;}
.c-medium {background: #fef3c7; color: #92400e;}
.c-hard {background: #fee2e2; color: #991b1b;}
.c-skill {background: #e0e7ff; color: #3730a3;}
.c-level {background: #ede9fe; color: #5b21b6; font-size: 1rem; padding: 6px 18px;}
.c-mode {background: #e2e8f0; color: #334155;}
.qtext {font-size: 1.1rem; font-weight: 600; line-height: 1.9; margin: 10px 0 4px;}
.atext {line-height: 1.95;}
</style>
"""


def card():
    try:
        return st.container(border=True)
    except TypeError:
        return st.container()


# ---------- Loaders (بيتحملوا مرة واحدة) ----------
@st.cache_resource(show_spinner=False)
def load_retriever():
    emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"})
    db = Chroma(persist_directory="chroma_db", embedding_function=emb)
    return db.as_retriever(search_kwargs={"k": 3})


@st.cache_resource(show_spinner=False)
def load_llm():
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16, device_map="auto")
    if os.path.isdir(ADAPTER):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval()
    return tok, model


def generate(cv, job, use_rag, use_ft, lang, retriever, tok, model, has_adapter):
    ctx = [d.page_content for d in retriever.invoke(f"{job} {cv}")] if use_rag else []
    ref = "\n".join(f"- {c}" for c in ctx) or "- (none)"
    if lang == "ar":
        lang_line = ("Output language: Arabic. Write question, skill_tested and ideal_answer in Arabic "
                     "(keep technical terms such as Django or SQL in English). "
                     "Keep candidate_level and difficulty values in English exactly as specified.")
    else:
        lang_line = "Output language: English."
    user = f"CV:\n{cv}\n\nJob: {job}\n\nReference material:\n{ref}\n\n{lang_line}"
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    cm = model.disable_adapter() if (has_adapter and not use_ft) else nullcontext()
    with cm, torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=1400, do_sample=True,
                             temperature=0.6, top_p=0.9)
    text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text, ctx


# ---------- Sidebar ----------
has_adapter = os.path.isdir(ADAPTER)
with st.sidebar:
    lang_choice = st.radio("اللغة / Language", ["العربية", "English"], horizontal=True)
    lang = "ar" if lang_choice == "العربية" else "en"
    t = TXT[lang]
    st.header(t["settings"])
    use_rag = st.toggle(t["rag"], value=True)
    use_ft = st.toggle(t["ft"], value=has_adapter, disabled=not has_adapter)
    if not has_adapter:
        st.caption("🔒 " + t["ft_missing"])

DIR = "rtl" if lang == "ar" else "ltr"
st.markdown(CSS, unsafe_allow_html=True)
st.markdown(f'<div class="hero" dir="{DIR}"><h1>🎯 {t["title"]}</h1><p>{t["sub"]}</p></div>',
            unsafe_allow_html=True)


def load_sample():
    st.session_state["cv_text"] = SAMPLE_CV


col_in, col_out = st.columns([1, 1.25], gap="large")

with col_in:
    with card():
        job = st.text_input(t["job"], "Backend Developer (Django)")
        pdf = st.file_uploader(t["upload"], type=["pdf"])
        st.text_area(t["paste"], height=230, key="cv_text")
        st.button(t["sample"], on_click=load_sample)
        run = st.button(t["go"], type="primary", use_container_width=True)

cv_text = st.session_state.get("cv_text", "")
if pdf is not None:
    cv_text = "\n".join((p.extract_text() or "") for p in PdfReader(pdf).pages)

with col_out:
    if run:
        cv = " ".join(cv_text.split())[:1200]
        if not cv:
            st.warning(t["empty"])
        else:
            with st.spinner(t["loading"]):
                retriever = load_retriever()
                tok, model = load_llm()
            with st.spinner(t["spinner"]):
                t0 = time.time()
                raw, ctx = generate(cv, job, use_rag, use_ft, lang, retriever, tok, model, has_adapter)
                secs = time.time() - t0
            try:
                result, info = parse_output(raw)
                st.session_state["out"] = {"result": result, "info": info, "ctx": ctx, "secs": secs,
                                           "job": job, "lang": lang, "rag": use_rag,
                                           "ft": bool(use_ft and has_adapter)}
            except Exception as ex:
                st.session_state["out"] = {"error": str(ex), "raw": raw}

    res = st.session_state.get("out")
    if res is None:
        st.info(t["hint"])
    elif "error" in res:
        st.error(t["fail"])
        with st.expander(t["raw"]):
            st.code(res["raw"])
    else:
        r = res["result"]
        mode = f'RAG {"✔" if res["rag"] else "✘"} · Fine-tuned {"✔" if res["ft"] else "✘"}'
        st.markdown(
            f'<div dir="{DIR}"><span class="chip c-level">{t["level"]}: {t["levels"][r.candidate_level]}</span>'
            f'<span class="chip c-mode">{mode}</span></div>', unsafe_allow_html=True)

        for i, q in enumerate(r.questions, 1):
            with card():
                st.markdown(
                    f'<div dir="{DIR}"><span class="chip c-{q.difficulty.lower()}">{t["diffs"][q.difficulty]}</span>'
                    f'<span class="chip c-skill">{html.escape(q.skill_tested)}</span>'
                    f'<div class="qtext">{t["q"]}{i}. {html.escape(q.question)}</div></div>',
                    unsafe_allow_html=True)
                with st.expander(t["answer"]):
                    st.markdown(rtl(q.ideal_answer, "atext", lang), unsafe_allow_html=True)

        st.download_button(t["dl"], to_text(r, lang, res["job"]), file_name="interview_questions.txt",
                           mime="text/plain", use_container_width=True)
        status = t["valid"] if not (res["info"]["repaired"] or res["info"]["normalized"]) else t["fixed"]
        st.caption(f'{status} · {t["time"]}: {res["secs"]:.0f} {t["sec"]}')
        if res["ctx"]:
            with st.expander(t["refs"]):
                for c in res["ctx"]:
                    st.write("•", c)
