# KABLAM :collision:

_Kanban goes boom_

## Description

Built a thing that's like codex and claude code, only it is built for editability. Since we're dealing with local models it aims to store snapshots be completely reversable, editable, and replayable. By this it means if we identify one of our prompts has led the model to have bad output we can delete or edit that prompt and see what the model now outputs. We may leverage git in the future to create a "safe space" for the LLM to dirty history (using checkouts and commits) so it can "time travel" after we make edits to the prompt history. So each llm tool call that mutates the workspace will force a commit to it's own independent branch. So if we edit a tool we will replace the entire git history for the commit before that tool calling action. 

Alo how possible local agents are for automating simple ticket solves, the core idea is to programatically checkout a branch, attempt a fix using local models, create a review, moving the ticket as work is done (programatically)


## Features

- [ ] checkpointing
- [ ] tool calls `mutate / query`
- [x] tabs + traversal (with arrow keys)
- [x] editable chat history
- [x] editable commands
- [x] models choices
- [x] local models
- [ ] frontier models
- [x] commands
- [x] threaded conversations
- [x] markdown parsing
- [x] code block syntax highlighting
- [ ] searching
- [x] navigation
- [x] vim keybindings
- [x] alerts (e.g * on a threaded name tab when an event has occured)


