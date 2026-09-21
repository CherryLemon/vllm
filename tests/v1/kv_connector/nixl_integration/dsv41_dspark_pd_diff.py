# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diagnostic: compare two PD passes and a standalone reference per prompt."""

import argparse
import json
import os

import openai

from test_dsv41_dspark_pd import MAX_TOKENS, MODEL_NAME, PROMPTS


def run(base_url: str, prompts: list[str]) -> list[str]:
    client = openai.OpenAI(api_key="EMPTY", base_url=base_url)
    out = []
    for prompt in prompts:
        resp = client.completions.create(
            model=MODEL_NAME,
            prompt=prompt,
            max_tokens=MAX_TOKENS,
            temperature=0.0,
            top_p=1.0,
            extra_body={"add_special_tokens": False},
        )
        c = resp.choices[0]
        out.append(c.text)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8192)
    ap.add_argument("--reference", default="")
    args = ap.parse_args()
    base = f"http://{args.host}:{args.port}/v1"
    prompts = PROMPTS
    ref = {}
    if args.reference and os.path.exists(args.reference):
        ref = json.load(open(args.reference))

    a = run(base, prompts)
    b = run(base, prompts)
    c = run(base, prompts)
    for i, p in enumerate(prompts):
        r = ref.get(p, "<no-ref>")
        flag = "OK " if (a[i] == b[i] == c[i]) else "DIFF"
        print(f"[{flag}] prompt#{i} {p[:52]!r}")
        print(f"    pass1={a[i][:90]!r}")
        print(f"    pass2={b[i][:90]!r}")
        print(f"    pass3={c[i][:90]!r}")
        print(f"    ref  ={r[:90]!r}")
        print(f"    match: p1ref={a[i]==r} p2ref={b[i]==r} p3ref={c[i]==r}")


if __name__ == "__main__":
    main()
