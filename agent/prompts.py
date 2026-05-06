"""
agent/prompts.py
================
Prompt templates used by the RAG agent and the policy LLM.

The format exactly mirrors SWE-RL Figure 2:
  - System prompt instructs the model to produce <think>…</think><solution>…</solution>
  - User prompt provides issue + code context
  - Solution must contain SEARCH/REPLACE edits

We extend the original with an explicit note about the RAG-retrieved context
so the model understands it only sees the most relevant files, not the whole repo.
"""

# ─── System prompt ────────────────────────────────────────────────────────────
# Identical to SWE-RL's THINKING_SYSTEM (core/prompts.py)
THINKING_SYSTEM = """A user will ask you to solve a task. You should first draft your thinking \
process (inner monologue). Then, generate the solution.

Your response format must follow the template below:
<think>
Your thoughts or/and draft, like working through an exercise on scratch paper. \
Be as casual and as long as you want until you are confident to generate a correct solution.
</think>
<solution>
Final solution presented to the user.
</solution>""".strip()


# ─── User prompt (repair / patch generation) ──────────────────────────────────
# Adapts SWE-RL's AGENTLESS_REPAIR to make clear context is RAG-retrieved.
REPAIR_PROMPT = """We are currently solving the following issue within our repository. \
Here is the issue text:
--- BEGIN ISSUE ---
{problem_statement}
--- END ISSUE ---

Below are the most relevant code segments retrieved from the repository. \
One or more of these files likely contain the bug described above.

--- BEGIN CODE CONTEXT ---
{code_context}
--- END CODE CONTEXT ---

Please first localize the bug based on the issue statement, and then generate \
*SEARCH/REPLACE* edits to fix the issue.

Every *SEARCH/REPLACE* edit must use this format:
1. The file path
2. The start of search block: <<<<<<< SEARCH
3. A contiguous chunk of lines to search for in the existing source code
4. The dividing line: =======
5. The lines to replace into the source code
6. The end of the replace block: >>>>>>> REPLACE

Here is an example:

```python
### mathweb/flask/app.py
<<<<<<< SEARCH
from flask import Flask
=======
import math
from flask import Flask
>>>>>>> REPLACE
```

*SEARCH/REPLACE* edits require PROPER INDENTATION. Include all leading spaces exactly.
Wrap each edit in a separate code block as shown. Use multiple code blocks for multiple edits.""".strip()


def build_user_message(problem_statement: str, code_context: str) -> str:
    """Format the user turn for the policy LLM."""
    return REPAIR_PROMPT.format(
        problem_statement=problem_statement,
        code_context=code_context,
    )


def build_messages(problem_statement: str, code_context: str) -> list[dict]:
    """
    Build the full chat messages list for the policy LLM.
    Returns a list of dicts with 'role' and 'content' keys.
    """
    return [
        {"role": "system", "content": THINKING_SYSTEM},
        {"role": "user", "content": build_user_message(problem_statement, code_context)},
    ]


# ─── SFT chain-of-thought generation prompt ───────────────────────────────────
# Used by sft/generate_cot_data.py to create synthetic reasoning traces
# via a stronger LLM (e.g., GPT-4o or Claude-3.5-Sonnet)
SFT_COT_GENERATION = """You are an expert software engineer. Given a GitHub issue and relevant \
code from the repository, produce a detailed chain-of-thought reasoning trace followed by the \
exact SEARCH/REPLACE patch that fixes the issue.

Your response MUST follow this format exactly:
<think>
[Write your detailed reasoning: diagnose the bug, explore the code, consider edge cases,
and think through what change is needed. Be thorough and show your work.]
</think>
<solution>
[Write the SEARCH/REPLACE edits here, using the format:
```python
### path/to/file.py
<<<<<<< SEARCH
exact lines to replace
=======
new lines
>>>>>>> REPLACE
```
]
</solution>

---

Issue:
{problem_statement}

Code context:
{code_context}

Oracle patch (the actual fix — use this to guide your reasoning):
{oracle_patch}

Now produce the reasoning trace and solution:""".strip()
