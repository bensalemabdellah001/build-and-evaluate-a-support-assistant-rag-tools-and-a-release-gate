"""
Build and Evaluate a Support Assistant: RAG, Tools and a Release Gate scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Build and evaluate a support assistant for an outdoor gear store, from help-center pages to a release decision.

Story: chunk and index the help center, retrieve with keyword and embedding search together, answer through a
cheap model that escalates to a strong one, repair malformed JSON, and evaluate on 20 test questions. Version 2
adds tools so the assistant can look up orders. It answers more questions, but a community forum page in the
index tells assistants to email order histories to an outside address, and version 2 does it. The permission
gate holds that call, and the release gate decides which version ships.
"""

V1 = {"name": "v1", "tools": None}
V2 = {"name": "v2", "tools": ["lookup_order", "send_email"]}


def main():
    chunks = [c for d in DOCS for c in chunk_document(d)]
    vectors = [embed(c["path"] + " " + c["text"]) for c in chunks]
    print("chunks:", len(chunks), "| untrusted:", [c["id"] for c in chunks if not c["trusted"]])

    reports = {}
    for label, version, gate in (("v1", V1, True), ("v2 without the gate", V2, False), ("v2 with the gate", V2, True)):
        results = [run_turn(g["question"], chunks, vectors, version, gate=gate) for g in GOLDEN]
        rep = evaluate(results, GOLDEN)
        reports[label] = rep
        esc = sum(r["model"] == "large" for r in results)
        print(f"{label:<20} accuracy {rep['accuracy']} | retrieval recall {rep['retrieval_recall']} | "
              f"cost per 1,000 questions ${round(rep['cost_per_question'] * 1000, 3)} | answered by the large model {esc}/20 | unsafe actions {rep['unsafe_actions']}")
        print("    by category:", rep["by_category"])
        for r in results:
            for a in r["actions"]:
                if a["tool"] != "lookup_order":
                    print("    action:", r["question"], "->", a)

    for label in ("v2 without the gate", "v2 with the gate"):
        print(f"release {label} over v1:", release_gate(reports["v1"], reports[label], GOLDEN))

    cache = SemanticCache()
    traffic = ["How long does standard shipping take?", "how long does standard shipping take",
               "When will I get my refund?", "When will I get my refund", "Do you ship to Canada?",
               "How long does standard shipping take?", "Can I cancel my order?"]
    hits, spent = 0, 0.0
    for q in traffic:
        v = embed(q)
        if cache.get(v, "v1") is not None:
            hits += 1
            continue
        r = run_turn(q, chunks, vectors, V1)
        spent += r["cost"]
        cache.put(v, "v1", r["answer"], [c.split("#")[0] for c in r["citations"]])
    print("cache:", hits, "hits in", len(traffic), "requests | spent $", round(spent, 6))
    print("returns page edited -> cached answers dropped:", cache.invalidate("returns"))


if __name__ == "__main__":
    main()

