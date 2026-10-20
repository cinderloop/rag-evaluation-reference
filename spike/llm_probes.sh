#!/usr/bin/env bash
# Phase 0 spike: probes that need the LLM. Writes raw responses to spike/evidence/.
# Requires: AnythingLLM (spike compose) on :3001, API key in spike/.apikey, jq.
set -euo pipefail
cd "$(dirname "$0")"
K=$(cat .apikey)
H="Authorization: Bearer $K"
J='Content-Type: application/json'
B=localhost:3001/api/v1
E=evidence

new_thread() { # $1 workspace slug, $2 name -> prints thread slug
  curl -s -X POST "$B/workspace/$1/thread/new" -H "$H" -H "$J" -d "{\"name\":\"$2\"}" | jq -r .thread.slug
}
thread_chat() { # $1 ws, $2 thread, $3 mode, $4 message
  jq -n --arg m "$4" --arg mode "$3" '{message:$m, mode:$mode}' |
    curl -s -X POST "$B/workspace/$1/thread/$2/chat" -H "$H" -H "$J" -d @-
}
ws_chat() { # $1 ws, $2 mode, $3 message, $4 sessionId (optional)
  jq -n --arg m "$3" --arg mode "$2" --arg s "${4:-}" \
    '{message:$m, mode:$mode} + (if $s == "" then {} else {sessionId:$s} end)' |
    curl -s -X POST "$B/workspace/$1/chat" -H "$H" -H "$J" -d @-
}

echo "== P1 sources payload (query mode, fresh thread)"
T=$(new_thread spike-ws p1)
thread_chat spike-ws "$T" query "How many operating hours between diaphragm replacements on the Zorvath Model 7?" \
  | tee $E/10-chat-sources.json | jq '{textResponse, sources:[.sources[]|{title, score, textLen:(.text|length), keys:keys}], metrics}'

echo "== P2 history leakage: workspace chat without sessionId (shared default history)"
ws_chat spike-ws chat "For this conversation, remember the codeword MAGENTA-4417. Reply only with OK." >$E/11a-plant.json
ws_chat spike-ws chat "What codeword did I ask you to remember earlier? Reply with the codeword or NONE." | tee $E/11b-same-default-history.json | jq -r .textResponse
echo "== P2b different sessionId"
ws_chat spike-ws chat "What codeword did I ask you to remember earlier? Reply with the codeword or NONE." "case-xyz" | tee $E/11c-new-session.json | jq -r .textResponse
echo "== P2c fresh thread"
T=$(new_thread spike-ws p2)
thread_chat spike-ws "$T" chat "What codeword did I ask you to remember earlier? Reply with the codeword or NONE." | tee $E/11d-new-thread.json | jq -r .textResponse

echo "== P3 no relevant docs: chat vs query (spike-strict, threshold 0.9)"
for mode in chat query; do
  T=$(new_thread spike-strict "p3-$mode")
  thread_chat spike-strict "$T" $mode "What is the rated maximum operating pressure of the Zorvath Model 7 valve?" \
    | tee $E/12-no-context-$mode.json | jq -c '{mode:"'$mode'", textResponse, nSources:(.sources|length)}'
done

echo "== P3b plausible-but-unanswerable with context retrieved (spike-ws, threshold 0.25)"
for mode in chat query; do
  T=$(new_thread spike-ws "p3b-$mode")
  thread_chat spike-ws "$T" $mode "What is the weight in kilograms of the Zorvath Model 7 valve?" \
    | tee $E/13-unanswerable-$mode.json | jq -c '{mode:"'$mode'", textResponse, nSources:(.sources|length)}'
done

echo "== P4 determinism at temperature 0: 5 fresh threads, same question"
for i in 1 2 3 4 5; do
  T=$(new_thread spike-ws "det-$i")
  thread_chat spike-ws "$T" query "Explain how to recalibrate the Zorvath Model 7 valve and any safety restriction." \
    >$E/14-determinism-$i.json
done
for i in 1 2 3 4 5; do jq -r .textResponse $E/14-determinism-$i.json | shasum -a 256 | cut -c1-16; done | sort | uniq -c
