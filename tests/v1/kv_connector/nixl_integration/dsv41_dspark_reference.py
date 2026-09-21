# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dump a standalone DSpark reference for the PD acceptance test.

Runs the same prompts as ``test_dsv41_dspark_pd.py`` against a *single*
(non-disaggregated) DSpark server at ``temperature=0`` and writes
``{prompt: text}`` JSON.  Point ``DSV41_DSPARK_REFERENCE`` at the result to
assert that the P/D path is transport-transparent.

    python3 dsv41_dspark_reference.py --port 8400 --out /tmp/ref.json
"""

import argparse
import json

import openai

from test_dsv41_dspark_pd import MAX_TOKENS, MODEL_NAME, PROMPTS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    client = openai.OpenAI(
        api_key="EMPTY", base_url=f"http://{args.host}:{args.port}/v1"
    )
    ref: dict[str, str] = {}
    for prompt in PROMPTS:
        resp = client.completions.create(
            model=MODEL_NAME,
            prompt=prompt,
            max_tokens=MAX_TOKENS,
            temperature=0.0,
            top_p=1.0,
            extra_body={"add_special_tokens": False},
        )
        ref[prompt] = resp.choices[0].text
        print(f"{prompt[:44]!r} -> {ref[prompt][:60]!r}")
    with open(args.out, "w") as fh:
        json.dump(ref, fh, indent=1)
    print(f"wrote {len(ref)} prompts -> {args.out}")


if __name__ == "__main__":
    main()
