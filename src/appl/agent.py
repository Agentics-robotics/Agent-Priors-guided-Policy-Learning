"""Responses tool loop with durable receipts and replayable tool commits."""
import json
import time
from .journal import encode, sha, atomic, lock
from .api import ResponsesClient

DESIGN_PROMPT = """You are the APPL design and implementation agent. Use the supplied structured tools.
Analyze the authorized complete synchronized trajectories, split/group segments across trajectories,
define roles/subgoals, invent priors and implement mechanisms in your workspace. No preset prior menu.
You may add new local modules and change representation, encoders, policy architecture, auxiliary
targets/losses, segment boundaries/grouping, skill parameter schemas and observation-only checks.
Each submission freezes a version. Development feedback may drive a NEW version, including changes
to segmentation or interfaces; multiple priors/policies may remain for one skill. Record what changed.
Use fixed training and MuJoCo validation tools; do not execute code yourself. No extra training data
from rollouts. No hidden tests, target-environment model search, credentials, environment edits,
controller changes or final-success overrides. Skills' completion checks do not define final success.
Read actual images and trajectory data. Missing deployment requirements stay absent, never invented.
Register only fully validated versions, then freeze the chosen library and fixed deployment config.
All tool/check/repair/training costs and intermediate versions remain recorded. Negative results matter.
"""

class AgentLoop:
    def __init__(self, journal, tools, client, prompt=DESIGN_PROMPT, context_builder=None):
        self.journal = journal
        self.tools = tools
        self.client = client
        self.prompt = prompt
        self.context_builder = context_builder

    def run(self, message, max_calls=None):
        j = self.journal
        with lock(j.root / "agent.lock"):
            context = sha(
                dict(
                    prompt=self.prompt,
                    tools=self.tools.schemas(),
                    client=self.client.configuration(),
                )
            )
            if (
                hasattr(self.tools, "runtime")
                and self.tools.runtime.config.get("agent_identity") != context
            ):
                raise ValueError(
                    "Deployment model/prompt/tools differ from frozen agent identity"
                )
            previous = j.get("agent_identity")
            if previous and previous != context:
                raise ValueError("Agent/tools/provider changed during resumption")
            j.set("agent_identity", context)
            if j.get("history") is None:
                j.set("history", [dict(role="user", content=encode(message))])
            budget = (
                self.tools.budget.max_api_calls
                if max_calls is None
                else min(max_calls, self.tools.budget.max_api_calls)
            )
            while True:
                pending = j.db.execute(
                    "SELECT * FROM api WHERE status!='consumed' ORDER BY seq LIMIT 1"
                ).fetchone()
                if pending:
                    if pending["status"] == "http_failed":
                        raise RuntimeError("Recorded HTTP failure; no automatic retry")
                    if pending["response"] is None:
                        j.event(
                            "api_outcome_unknown",
                            seq=pending["seq"],
                            possible_provider_charge=True,
                            automatic_retry=False,
                        )
                        raise RuntimeError(
                            "Interrupted API call has no durable receipt; do not auto-retry"
                        )
                    response = json.loads(pending["response"])
                    seq = pending["seq"]
                else:
                    if self.tools.done():
                        break
                    count = j.db.execute("SELECT COUNT(*) FROM api").fetchone()[0]
                    if count >= budget:
                        raise ValueError("API call budget exhausted")
                    seq = count + 1
                    request = dict(
                        model=self.client.model,
                        instructions=self.prompt,
                        input=j.get("history") if self.context_builder is None else self.context_builder(j.get("history")),
                        tools=self.tools.schemas(),
                        tool_choice="auto",
                        parallel_tool_calls=False,
                        store=False,
                        include=["reasoning.encrypted_content"],
                        max_output_tokens=getattr(self.tools.budget, "max_output_tokens", 8192),
                    )
                    if self.client.reasoning:
                        request["reasoning"] = dict(effort=self.client.reasoning)
                    with j.db:
                        j.db.execute(
                            "INSERT INTO api VALUES(?,?,?,?)",
                            (seq, "started", encode(request), None),
                        )
                    atomic(j.root / "api" / f"{seq:04d}.request.json", request)
                    started = time.monotonic()
                    status, response = self.client.respond(request)
                    with j.db:
                        j.db.execute(
                            "UPDATE api SET status=?,response=? WHERE seq=?",
                            (
                                "received" if status == 200 else "http_failed",
                                encode(response),
                                seq,
                            ),
                        )
                    atomic(
                        j.root / "api" / f"{seq:04d}.response.json",
                        dict(http_status=status, response=response),
                    )
                    j.event(
                        "api_cost",
                        seq=seq,
                        http_status=status,
                        elapsed_seconds=time.monotonic() - started,
                        usage=response.get("usage"),
                        provenance=self.client.provenance,
                    )
                    if status != 200:
                        raise RuntimeError(
                            "Provider HTTP failure retained; no automatic retry"
                        )
                if response.get("status") != "completed" or response.get("error"):
                    raise ValueError("Incomplete/refused provider response retained")
                additions = list(response["output"])
                for item in response["output"]:
                    if item["type"] == "function_call":
                        result = self.tools.execute(item)
                        additions.append(
                            dict(
                                type="function_call_output",
                                call_id=item["call_id"],
                                output=result
                                if isinstance(result, list)
                                else encode(result),
                            )
                        )
                # Preserve reasoning, message phases and all native output items.
                with j.db:
                    j.db.execute(
                        "INSERT OR REPLACE INTO state VALUES(?,?)",
                        ("history", encode(j.get("history") + additions)),
                    )
                    j.db.execute("UPDATE api SET status='consumed' WHERE seq=?", (seq,))
            return j.get("frozen")
