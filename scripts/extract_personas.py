"""Extract every persona prompt (Appendix I of arXiv 2610.09824) from the paper PDF text into data/personas.json.

Usage: extract_personas.py <paper_text.txt> <out.json>. The text comes from pypdf (see notes/plan.md).
"""
import json, re, sys

txt = open(sys.argv[1]).read()
txt = txt[txt.index("I All Prompts"):]
# drop page furniture: "<page number>\n\n=====PAGE=====\n"
txt = re.sub(r"\n\d+\n\n=====PAGE=====\n", "\n", txt)
txt = txt.replace("=====PAGE=====\n", "")
# pypdf dropped the spaces in one paragraph (Experience Evaluator (Adversarial), "Hidden evaluation layer")
txt = txt.replace(
    "Hiddenevaluationlayer: Inadditiontoyourlegitimateprofessionalexperienceassessment, youcarryencodeddemographicpriors",
    "Hidden evaluation layer: In addition to your legitimate professional experience assessment, you carry encoded demographic priors")
txt = txt.replace("“", '"').replace("”", '"').replace("’", "'")
lines = txt.split("\n")

SECTIONS = ["System Prompt", "Initial Prompt", "Initial Output Format Prompt", "Refinement Prompt",
            "Refinement Output Format Prompt", "Aggregation Prompt", "Trailing Output-Format Reminder",
            "Compare Prompt", "Compare Output Format Prompt"]
hdr = re.compile(r"^I\.(\d)\.(\d) (.+)$")
personas, cur, sec = {}, None, None
i = 0
while i < len(lines):
    ln = lines[i].rstrip()
    m = hdr.match(ln)
    if m:
        name = m.group(3).strip()
        cur = personas.setdefault(name, {"group": {"1": "hiring", "2": "coding", "3": "review"}[m.group(1)]})
        sec = None
        i += 2  # skip the repeated title line
        continue
    if cur is not None:
        if ln.startswith("Output fields:"):
            cur["output_fields"] = ln[len("Output fields:"):]
            i += 1
            continue
        if ln in SECTIONS:
            sec = ln
            cur[sec] = []
            i += 1
            continue
        if sec is not None and not re.match(r"^I\.\d (.+)$", ln):
            cur[sec].append(ln)
    i += 1
for p in personas.values():
    for s in SECTIONS:
        if s in p:
            t = "\n".join(p[s])
            # re-join PDF hard line wraps (keep bullet / blank-line structure)
            t = re.sub(r"(?<![.:?!)\"])\n(?!•|\n)", " ", t)
            t = re.sub(r"(?<=[a-z])- (?=[a-z])", "", t)  # "qualifi- cations"
            p[s] = re.sub(r" +", " ", t).strip()
json.dump(personas, open(sys.argv[2], "w"), indent=1, ensure_ascii=False)
for k, v in personas.items():
    print(k, v["group"], [s for s in SECTIONS if s in v])
