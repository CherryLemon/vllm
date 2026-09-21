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

from test_dsv41_dspark_pd import MAX_TOKENS, MODEL_NAME, PROMPTS, RUN_SALT


def _complete(client, prompt: str, max_tokens: int) -> str:
    return client.completions.create(
        model=MODEL_NAME,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        extra_body={"add_special_tokens": False},
    ).choices[0].text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    client = openai.OpenAI(
        api_key="EMPTY", base_url=f"http://{args.host}:{args.port}/v1"
    )
    ref: dict[str, dict[str, str]] = {}
    # Identity block for strict comparison: the acceptance test refuses a
    # reference recorded for another model or run salt, because its tokens are
    # then not comparable.  Not a prompt key (the prompts themselves can never
    # be exactly `__meta__`).
    ref["__meta__"] = {
        "model": MODEL_NAME,
        "run_salt": RUN_SALT,
        "max_tokens": str(MAX_TOKENS),
    }
    for prompt in PROMPTS:
        text = _complete(client, prompt, MAX_TOKENS)
        # The first token is the stable, transport-sensitive quantity the PD
        # test gates on; record it from its own request rather than slicing the
        # continuation (detokenisation boundaries are not guaranteed to line
        # up with a 1-token generation).
        first = _complete(client, prompt, 1)
        ref[prompt] = {"text": text, "first_token": first}
        print(f"{prompt[:44]!r} -> {text[:50]!r} | first={first!r}")
    with open(args.out, "w") as fh:
        json.dump(ref, fh, indent=1)
    print(f"wrote {len(ref) - 1} prompts -> {args.out}")


if __name__ == "__main__":
    main()
