# explorer

A research agent built on LangGraph. Given a question, it searches the web,
fetches the pages, takes notes with source attribution and writes a cited
report. It shows every decision made along the way.

```
run-explorer research "How do free LLM API tiers compare in 2026?"

```
![explorer researching a question](assets/demo.gif)

explorer is a research agent with four tools at its disposal (set_plan, web_search,
fetch_page, save_note). The LLM takes the non-deterministic actions such as deciding which
tool to call, synthesising, writing the report while the graph binds it with
constraints to stop it going rogue on unbounded loops, drifting from its original
intent and producing nothing.

It is bound by the max turns (12) it can take, after which it must produce. At each turn
it is reminded of the state and position it is in: how many turns are left, how many
notes it has saved and which sub-questions it has left unanswered. If it tries to
fetch a page it has already read, the fetch is refused. If it tries to search again
while it still has unread results in hand, the search is refused and it is shown the
URLs it hasn't opened yet. And when it tries to finish with nothing in its notes, or
with sub-questions still unanswered, it is nudged back to work as long as it has
turns left to spend.


## What I learnt

I built this to understand the workflow of an agent, trace its path and understand why it takes the actions it takes. An interesting thing to be reminded of was an LLM's inability to hold any sense of position or state. It kept wanting to finish without producing an output, and had to be nudged back into action. During the first run, saving notes from the web pages it read wasn't bound by a rule, only asked for in the prompt, so it drifted further and further from the original question and ended up with nothing to report. It wasn't until the rules were enforced by the graph to make the agent stay on track. This made me realise how unreliable models can be at bookkeepings and how there has to be rules in place to make them reliable.

## Running it

<img src="assets/explorer.png" alt="The run-explorer command line" width="560">