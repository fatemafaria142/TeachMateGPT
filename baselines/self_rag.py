"""Self-RAG baseline — prompted reflection, as a *verifier*.

Asai et al., *Self-RAG: Learning to Retrieve, Generate, and Critique through
Self-Reflection* (ICLR 2024).

**Important caveat, to be stated in the paper caption and body:** real Self-RAG
trains a critic to emit reflection tokens during decoding. This implementation
prompts an off-the-shelf model to produce the same judgements post-hoc. It is
therefore *Self-RAG-style (prompted, not trained critic)*. Claiming otherwise is
a rejection risk; disclosing it costs nothing.

Reflection tokens implemented:

* ``ISREL``  — is the retrieved evidence relevant?
* ``ISSUP``  — is the generated item supported by that evidence?
* ``ISUSE``  — is the item useful (1-5)?

Retrieval is held fixed at the full hybrid pipeline so that only the verification
policy differs from SAVER.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

_ISREL_PROMPT = (
    "প্রশ্ন: {query}\n\nউদ্ধৃত পাঠ্যাংশ:\n{context}\n\n"
    "পাঠ্যাংশটি কি প্রশ্নের জন্য প্রাসঙ্গিক? ISREL=relevant অথবা ISREL=irrelevant দাও।\n\nউত্তর:"
)
_ISSUP_PROMPT = (
    "উদ্ধৃত পাঠ্যাংশ:\n{context}\n\nতৈরি হওয়া প্রশ্ন/উত্তর:\n{item}\n\n"
    "এটি কি পাঠ্যাংশ দ্বারা সমর্থিত? ISSUP=fully / partially / no দাও।\n\nউত্তর:"
)
_ISUSE_PROMPT = (
    "শিক্ষার্থীর জন্য নিচের মূল্যায়ন আইটেমটি কতটা উপযোগী? ISUSE=1..5 দাও।\n\n{item}\n\nউত্তর:"
)


@dataclass
class ReflectionVerdict:
    isrel: str = "unknown"
    issup: str = "unknown"
    isuse: Optional[int] = None
    accepted: bool = True
    llm_calls: int = 0
    raw: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "isrel": self.isrel, "issup": self.issup, "isuse": self.isuse,
            "accepted": self.accepted, "llm_calls": self.llm_calls,
        }


class SelfRagVerifier:
    name = "self_rag"

    def __init__(self, llm, *, min_isuse: int = 3, require_support: bool = True) -> None:
        self.llm = llm
        self.min_isuse = min_isuse
        self.require_support = require_support

    def verify(self, query: str, contexts: Sequence[str], item_text: str) -> ReflectionVerdict:
        ctx = "\n\n".join(contexts)[:4000] or "(কোনো প্রমাণ পাওয়া যায়নি)"
        calls = 0

        rel_raw = str(self.llm.invoke(_ISREL_PROMPT.format(query=query, context=ctx)))
        calls += 1
        isrel = "relevant" if "irrelevant" not in rel_raw.lower() and "relevant" in rel_raw.lower() else (
            "irrelevant" if "irrelevant" in rel_raw.lower() else "unknown")

        sup_raw = str(self.llm.invoke(_ISSUP_PROMPT.format(context=ctx, item=item_text[:2000])))
        calls += 1
        low = sup_raw.lower()
        issup = "fully" if "fully" in low else ("partially" if "partial" in low else
                                                ("no" if re.search(r"\bno\b", low) else "unknown"))

        use_raw = str(self.llm.invoke(_ISUSE_PROMPT.format(item=item_text[:2000])))
        calls += 1
        m = re.search(r"[1-5]", use_raw)
        isuse = int(m.group(0)) if m else None

        accepted = True
        if isrel == "irrelevant":
            accepted = False
        if self.require_support and issup == "no":
            accepted = False
        if isuse is not None and isuse < self.min_isuse:
            accepted = False

        return ReflectionVerdict(
            isrel=isrel, issup=issup, isuse=isuse, accepted=accepted,
            llm_calls=calls,
            raw={"isrel": rel_raw[:200], "issup": sup_raw[:200], "isuse": use_raw[:200]},
        )
