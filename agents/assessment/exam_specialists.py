"""
Format-specific assessment specialists (MCQ, সৃজনশীল, …).

Implements one prompt path per task type: loads the right template, calls the LLM
with the grounded context blob, parses JSON, and validates format, language, and
basic grounding before returning capped lists sized by ``ASSESSMENT_COUNTS``.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, List

import config
from utils.llm_client import chat_completion
from utils.prompt_loader import load_prompt


def _context_blob(context: List[str]) -> str:
    return " ".join(context) if context else ""


def _parse_slice(result_str: str, key: str) -> List[Any]:
    try:
        data = json.loads(result_str)
        if not isinstance(data, dict):
            return []
        v = data.get(key)
        return v if isinstance(v, list) else []
    except Exception:
        return []


def _bangla_ratio(text: str) -> float:
    t = text or ""
    letters = [ch for ch in t if ch.isalpha()]
    if not letters:
        return 1.0
    bn = sum(1 for ch in letters if "\u0980" <= ch <= "\u09ff")
    return bn / max(1, len(letters))


def _has_non_bangla_letters(text: str) -> bool:
    for ch in text or "":
        if ch.isalpha() and not ("\u0980" <= ch <= "\u09ff"):
            return True
    return False


def _is_bangla_only_enough(text: str, *, min_ratio: float = 0.88) -> bool:
    if _has_non_bangla_letters(text):
        return False
    return _bangla_ratio(text) >= min_ratio


def _is_bangla_mcq_text_enough(text: str, *, min_ratio: float = 0.55) -> bool:
    """
    MCQ stems/options can contain a few scientific terms in Latin script
    (e.g., Chordata, Ascaris). Keep Bangla dominant instead of hard-rejecting.
    """
    return _bangla_ratio(text) >= min_ratio


def _parse_answer_index(value: Any) -> int | None:
    if isinstance(value, int):
        return value if 0 <= value <= 3 else None
    if value is None:
        return None
    s = str(value).strip().lower()
    if not s:
        return None
    bangla_digits = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
    s = s.translate(bangla_digits)
    direct = {
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "ক": 0,
        "খ": 1,
        "গ": 2,
        "ঘ": 3,
        "ক)": 0,
        "খ)": 1,
        "গ)": 2,
        "ঘ)": 3,
        "ক.": 0,
        "খ.": 1,
        "গ.": 2,
        "ঘ.": 3,
    }
    return direct.get(s)


def _normalize_mcq_item(item: Any) -> Any:
    if not isinstance(item, dict):
        return item
    out = dict(item)
    parsed = _parse_answer_index(out.get("answer_index"))
    if parsed is None and "answer" in out:
        parsed = _parse_answer_index(out.get("answer"))
    out["answer_index"] = parsed
    return out


def _is_mcq_bangla_valid(item: Any) -> bool:
    item = _normalize_mcq_item(item)
    if not isinstance(item, dict):
        return False
    q = str(item.get("question") or "")
    opts = item.get("options") or []
    if not q or not isinstance(opts, list) or len(opts) != 4:
        return False
    ans = item.get("answer_index")
    if not isinstance(ans, int) or ans < 0 or ans > 3:
        return False
    if not _is_bangla_mcq_text_enough(q):
        return False
    for op in opts:
        if not _is_bangla_mcq_text_enough(str(op or "")):
            return False
    return True


def _is_mcq_grounded_enough(item: Any, context: List[str]) -> bool:
    """
    Lightweight grounding check: ensure the MCQ reuses important tokens
    from the retrieved textbook snippets instead of hallucinating topics.
    """
    if not isinstance(item, dict):
        return False
    q = str(item.get("question") or "")
    opts = item.get("options") or []
    if not q or not isinstance(opts, list):
        return False
    ctx_blob = " ".join(context[:4])
    ctx_tokens = _normalized_tokens(ctx_blob)
    if not ctx_tokens:
        return True
    q_tokens = _normalized_tokens(q)
    opt_tokens = set()
    for op in opts:
        opt_tokens |= _normalized_tokens(str(op or ""))
    focus_tokens = q_tokens | opt_tokens
    if not focus_tokens:
        return False
    overlap = _jaccard(focus_tokens, ctx_tokens)
    return overlap >= 0.08


def _is_cq_bangla_valid(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    headline = str(item.get("headline") or "")
    clue = str(item.get("clue") or "")
    if headline and not _is_bangla_only_enough(headline):
        return False
    if not _is_bangla_only_enough(clue):
        return False
    parts = item.get("parts") or []
    if not isinstance(parts, list):
        return False
    for p in parts:
        if not isinstance(p, dict):
            return False
        if not _is_bangla_only_enough(str(p.get("question") or "")):
            return False
        if not _is_bangla_only_enough(str(p.get("skill_bn") or "")):
            return False
    return True


def _uniq_key_mcq(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    return (
        str(item.get("question") or "").strip().lower()
        + "||"
        + "|".join(str(x).strip().lower() for x in (item.get("options") or []))
    )


def _uniq_key_cq(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    clue = str(item.get("clue") or "").strip().lower()
    h = str(item.get("headline") or "").strip().lower()
    return h + "||" + clue


def _run(
    prompt_file: str,
    key: str,
    context: List[str],
    difficulty_level: str,
    target_count: int,
    extra_instructions: str = "",
) -> List[Any]:
    system_prompt = load_prompt(prompt_file)
    lvl = (difficulty_level or "intermediate").strip().lower()
    if lvl not in getattr(config, "DIFFICULTY_LEVELS", ()):
        lvl = "intermediate"
    label = getattr(config, "LEVEL_LABELS_BN", {}).get(lvl, lvl)
    user_msg = (
        f"প্রশ্নের স্তর: {label} (key: {lvl}).\n"
        f"JSON key `{key}` এর অ্যারেতে **ঠিক {target_count}টি** বস্তু থাকতে হবে। "
        f"কম বা বেশি দেওয়া যাবে না।\n\n"
        "ভাষা গার্ডরেইল: ছাত্র-দেখার সকল টেক্সট ১০০% বাংলা স্ক্রিপ্টে লিখবে। "
        "ইংরেজি বাক্য/ফ্রেজ/হাইব্রিড ভাষা দেবে না।\n"
        f"{extra_instructions.strip()}\n\n"
        f"Context:\n{_context_blob(context)}"
    )
    result_str = chat_completion(system_prompt, user_msg, json_mode=True)
    return _parse_slice(result_str, key)


def _collect_until_target(
    *,
    prompt_file: str,
    key: str,
    context: List[str],
    difficulty_level: str,
    target_count: int,
    validator: Callable[[Any], bool],
    uniq_key: Callable[[Any], str],
    max_attempts: int,
    extra_instructions: str = "",
) -> List[Any]:
    collected: List[Any] = []
    seen: set[str] = set()
    attempts = max(1, max_attempts)
    for _ in range(attempts):
        remaining = max(0, target_count - len(collected))
        if remaining <= 0:
            break
        batch = _run(
            prompt_file,
            key,
            context,
            difficulty_level,
            remaining,
            extra_instructions=extra_instructions
            + f"\nশুধু বাকি {remaining}টি নতুন আইটেম দাও; আগেরটার পুনরাবৃত্তি কোরো না।",
        )
        for item in batch:
            if not validator(item):
                continue
            k = uniq_key(item)
            if not k or k in seen:
                continue
            seen.add(k)
            collected.append(item)
            if len(collected) >= target_count:
                break
    return collected[:target_count]


def _sentence_count(text: str) -> int:
    parts = re.split(r"[।!?]+", (text or "").strip())
    return len([p for p in parts if p.strip()])


def _normalized_tokens(text: str) -> set[str]:
    tokens = re.findall(r"[\u0980-\u09ffa-zA-Z0-9]+", (text or "").lower())
    stop = {"এবং", "অথবা", "করে", "থেকে", "জন্য", "একটি", "উদ্দীপক", "হলো", "নিয়ে", "নিয়ে"}
    return {t for t in tokens if len(t) > 2 and t not in stop}


def _query_focus_terms(query_text: str) -> set[str]:
    q = _normalized_tokens(query_text or "")
    stop = {
        "শিক্ষার্থীরা",
        "অনুশীলনের",
        "বইয়ের",
        "আলোকে",
        "ভিত্তিক",
        "প্রশ্ন",
        "বিশ্লেষণধর্মী",
        "তৈরি",
        "ক্লাসে",
        "আলোচনা",
    }
    out = [t for t in q if t not in stop]
    return set(list(out)[:8])


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def _make_last_resort_mcq(seed_text: str, idx: int) -> dict:
    seed = re.sub(r"\s+", " ", (seed_text or "").strip())
    if len(seed) > 120:
        seed = seed[:120].rstrip() + "..."
    if not seed:
        seed = "পাঠ্যাংশের মূল ধারণা"
    return {
        "question": f"পাঠ্যাংশ অনুযায়ী নিচের কোন বক্তব্যটি সঠিক? (MCQ {idx})",
        "options": [
            f"ক) {seed}",
            "খ) পাঠ্যাংশে এর বিপরীত তথ্য দেওয়া হয়েছে",
            "গ) পাঠ্যাংশে এটি উল্লেখ করা হয়নি",
            "ঘ) সবগুলো সঠিক",
        ],
        "answer_index": 0,
    }


def _force_fill_mcqs(out: List[Any], context: List[str], target_count: int) -> List[Any]:
    normalized = [_normalize_mcq_item(x) for x in out if isinstance(x, dict)]
    normalized = [x for x in normalized if _is_mcq_bangla_valid(x)]
    seen = {_uniq_key_mcq(x) for x in normalized}
    seed_text = " ".join((context or [])[:2])
    i = 1
    while len(normalized) < target_count:
        candidate = _make_last_resort_mcq(seed_text, i)
        i += 1
        k = _uniq_key_mcq(candidate)
        if not k or k in seen:
            continue
        seen.add(k)
        normalized.append(candidate)
    return normalized[:target_count]


def _force_fill_cq(out: List[Any], context: List[str], target_count: int, *, query_text: str = "") -> List[Any]:
    valid = [x for x in out if isinstance(x, dict) and _is_cq_bangla_valid(x)]
    if len(valid) >= target_count:
        return valid[:target_count]
    seed = re.sub(r"\s+", " ", " ".join((context or [])[:2]).strip())
    if len(seed) > 140:
        seed = seed[:140].rstrip() + "..."
    if not seed:
        seed = "পাঠ্যাংশে আলোচিত বৈজ্ঞানিক ধারণা"
    focus_terms = sorted(_query_focus_terms(query_text))
    focus_phrase = ", ".join(focus_terms[:3]) if focus_terms else seed
    fallback = {
        "headline": "সৃজনশীল প্রশ্ন — পাঠভিত্তিক",
        "clue": (
            "উদ্দীপক:\n\n"
            f"শ্রেণিকক্ষে শিক্ষক আজকের পাঠে {focus_phrase} বিষয়ে আলোচনা করেছেন। "
            "শিক্ষার্থীরা বিষয়টি বুঝতে উদাহরণ, বৈশিষ্ট্য ও প্রয়োগ নিয়ে দলগত আলোচনা করে। "
            "আলোচনার শেষে শিক্ষক তাদের পর্যবেক্ষণ লিখতে বলেন এবং বাস্তব জীবনের সাথে মিলিয়ে ব্যাখ্যা করতে বলেন।"
        ),
        "parts": [
            {"label": "ক", "skill_bn": "জ্ঞানমূলক", "question": f"উদ্দীপকে আলোচিত {focus_phrase} ধারণার মৌলিক অর্থ কী?"},
            {"label": "খ", "skill_bn": "অনুধাবনমূলক", "question": f"{focus_phrase} ধারণা বংশগত বৈশিষ্ট্য বোঝার জন্য কেন গুরুত্বপূর্ণ?"},
            {"label": "গ", "skill_bn": "প্রয়োগমূলক", "question": f"উদ্দীপকের আলোকে {focus_phrase} ধারণার একটি পাঠ্যবইভিত্তিক প্রয়োগ ব্যাখ্যা করো।"},
            {"label": "ঘ", "skill_bn": "উচ্চতর দক্ষতা", "question": f"উদ্দীপক বিশ্লেষণ করে দেখাও, {focus_phrase} বিষয়ে ধারণাগত ভুল কমাতে কী ধরনের যুক্তিভিত্তিক ব্যাখ্যা দরকার।"},
        ],
    }
    if _is_cq_query_aligned(fallback, query_text):
        valid.append(fallback)
    return valid[:target_count]


def _is_direct_theory_dump(clue_text: str) -> bool:
    t = (clue_text or "").strip()
    patterns = [
        r"^উদ্দীপক:\s*\n*\s*(মাইটোসিস|মিয়োসিস|নিউটনের|সালোকসংশ্লেষণ|অভিকর্ষ|ঘর্ষণ|কোষ বিভাজন)\s+(হলো|বলতে|হয়)",
        r"(সংজ্ঞা|সূত্র|তত্ত্ব)\s+(অনুযায়ী|অনুযায়ী)",
    ]
    return any(re.search(p, t, flags=re.IGNORECASE) for p in patterns)


def _is_narrative_uddipok(item: Any, context: List[str]) -> bool:
    if not isinstance(item, dict):
        return False
    clue = str(item.get("clue") or "").strip()
    if not clue.startswith("উদ্দীপক:"):
        return False
    if _sentence_count(clue.replace("উদ্দীপক:", "", 1).strip()) < 5:
        return False
    if _is_direct_theory_dump(clue):
        return False
    ctx_tokens = _normalized_tokens(" ".join(context[:3]))
    clue_tokens = _normalized_tokens(clue)
    if _jaccard(clue_tokens, ctx_tokens) > 0.62:
        return False
    return True


def _is_cq_query_aligned(item: Any, query_text: str) -> bool:
    if not isinstance(item, dict):
        return False
    q_terms = _query_focus_terms(query_text)
    if not q_terms:
        return True
    clue = str(item.get("clue") or "")
    parts = item.get("parts") or []
    if not isinstance(parts, list):
        return False
    all_text = clue + " " + " ".join(str(p.get("question") or "") for p in parts if isinstance(p, dict))
    all_tokens = _normalized_tokens(all_text)
    return len(q_terms & all_tokens) >= min(2, len(q_terms))


def specialist_srijonshil(
    context: List[str],
    difficulty_level: str,
    target_count: int,
    *,
    query_text: str = "",
) -> List[Any]:
    extra = (
        "উদ্দীপক অবশ্যই বাস্তব জীবনের clue-based গল্প হবে; "
        "বইয়ের বাক্য কপি বা সরাসরি সংজ্ঞা/তত্ত্ব দিয়ে শুরু করা যাবে না। "
        "উদ্দীপকে চূড়ান্ত concept নাম সরাসরি বলে দিও না; পর্যবেক্ষণভিত্তিক ইঙ্গিত দাও।"
    )
    retry_extra = (
        extra
        + " এবার আগের আউটপুটের মত বই-ভিত্তিক বাক্য দেবে না। "
        + "বাংলাদেশি বাস্তব দৃশ্য (স্কুল/বাড়ি/খামার/ল্যাব/খেলাধুলা) ব্যবহার করো, ৫-৮ বাক্য লিখো।"
    )
    focus_terms = sorted(_query_focus_terms(query_text))
    focus_line = (
        (" Query concept বাধ্যতামূলক: " + ", ".join(focus_terms) + " — "
         "উদ্দীপক এবং গ/ঘ প্রশ্নে এগুলোর সাথে সরাসরি সম্পর্ক থাকতে হবে; generic লেখা যাবে না।")
        if focus_terms
        else ""
    )

    def _valid_cq(x: Any) -> bool:
        return (
            _is_narrative_uddipok(x, context)
            and _is_cq_bangla_valid(x)
            and _is_cq_query_aligned(x, query_text)
        )

    out = _collect_until_target(
        prompt_file="exam_srijonshil.md",
        key="creative_questions",
        context=context,
        difficulty_level=difficulty_level,
        target_count=target_count,
        validator=_valid_cq,
        uniq_key=_uniq_key_cq,
        max_attempts=int(getattr(config, "ASSESSMENT_CQ_MAX_ATTEMPTS", 5)),
        extra_instructions=retry_extra
        + focus_line
        + " Bangla-only strict mode: কোনো ইংরেজি/জাপানি/চাইনিজ/অন্যান্য স্ক্রিপ্ট ব্যবহার করবে না।",
    )
    return _force_fill_cq(out, context, target_count, query_text=query_text)


def specialist_mcqs(
    context: List[str],
    difficulty_level: str,
    target_count: int,
    *,
    query_text: str = "",
) -> List[Any]:
    _ = query_text
    def _valid_mcq(x: Any) -> bool:
        y = _normalize_mcq_item(x)
        return _is_mcq_bangla_valid(y) and _is_mcq_grounded_enough(y, context)

    def _append_new(collected: List[Any], batch: List[Any], validator: Callable[[Any], bool]) -> List[Any]:
        seen = {_uniq_key_mcq(x) for x in collected if isinstance(x, dict)}
        out = list(collected)
        for item in batch:
            if not validator(item):
                continue
            k = _uniq_key_mcq(item)
            if not k or k in seen:
                continue
            seen.add(k)
            out.append(item)
            if len(out) >= target_count:
                break
        return out

    out = _collect_until_target(
        prompt_file="exam_mcq.md",
        key="mcqs",
        context=context,
        difficulty_level=difficulty_level,
        target_count=target_count,
        validator=_valid_mcq,
        uniq_key=_uniq_key_mcq,
        max_attempts=int(getattr(config, "ASSESSMENT_MCQS_MAX_ATTEMPTS", 5)),
        extra_instructions=(
            "Bangla-only strict mode: কোনো ইংরেজি/জাপানি/চাইনিজ/অন্যান্য স্ক্রিপ্ট ব্যবহার করবে না।"
        ),
    )

    remaining = max(0, target_count - len(out))
    if remaining > 0:
        relaxed = _collect_until_target(
            prompt_file="exam_mcq.md",
            key="mcqs",
            context=context,
            difficulty_level=difficulty_level,
            target_count=remaining,
            validator=lambda x: _is_mcq_bangla_valid(_normalize_mcq_item(x)),
            uniq_key=_uniq_key_mcq,
            max_attempts=max(1, int(getattr(config, "ASSESSMENT_MCQS_MAX_ATTEMPTS", 5)) // 2),
            extra_instructions=(
                "শুধু বইভিত্তিক, কাঠামো-সঠিক MCQ দাও। "
                "answer_index অবশ্যই 0-3 হবে এবং options ঠিক 4টি হবে।"
            ),
        )
        out = _append_new(
            [_normalize_mcq_item(x) for x in out if isinstance(x, dict)],
            [_normalize_mcq_item(x) for x in relaxed if isinstance(x, dict)],
            lambda x: _is_mcq_bangla_valid(_normalize_mcq_item(x)),
        )

    remaining = max(0, target_count - len(out))
    if remaining > 0:
        raw = _run(
            "exam_mcq.md",
            "mcqs",
            context,
            difficulty_level,
            remaining,
            extra_instructions=(
                f"Final retry: বাকি ঠিক {remaining}টি নতুন MCQ দাও। "
                "বৈধ JSON, 4 options, answer_index 0-3 নিশ্চিত করো। "
                "শুধু পাঠ্যবইয়ের তথ্য ব্যবহার করো।"
            ),
        )
        out = _append_new(
            [_normalize_mcq_item(x) for x in out if isinstance(x, dict)],
            [_normalize_mcq_item(x) for x in raw if isinstance(x, dict)],
            lambda x: _is_mcq_bangla_valid(_normalize_mcq_item(x)),
        )

    return _force_fill_mcqs(out, context, target_count)


SPECIALIST_ORDER = [
    ("mcqs", specialist_mcqs),
    ("creative_questions", specialist_srijonshil),
]
