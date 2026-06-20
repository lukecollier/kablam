# Features


## Cmd failures

Command executions can potentially fail. We will surface this by turning the cmd pallete to red and displaying an error message above. The error message box is positioned in a way like the auto complete box but is above instead of below (so it's the same width of the command prompt). So for example if we tried to select a `<model-id>` that doesn't exist e.g `:model notavalidid` would display an error message above the box `notavalidid is not a recognised model, use :model ls to show available models`. We stay in the cmd window until a we send a valid cmd or we press escape to return to normal mode.

## Threads

Threads will be tabs in the history, we can navigate between the tabs by being in normal mode and pressing tab. it will go between the tabs. There will also be a command that allows rename tabs, using the number (1,2,3 etc) to select those tabs so `:tab 1` would go to the first tab in the list. Tabs can also have names (defaults to the current model) we can start a new tab witha specified model using `:tab new <model-id>` if we already have a tab with the same model the name should become <model-id><models-of-same-id+1> so if we have a tab for openllm2 we would change the previous tab's name to `openllm2-1` and then have a new tab called `openllm2-2`. This naming is done by having a `name-override` which is an option, if a user has selected a name using `:tab rename <tab-id|tab-name> <new-tab-name>` the command should fail if the tab name already exists. If the `name-override` isn't set we use the rule for making the names based of the `<model-id>`. We can also kill tabs with the `:tab kill <tab-name|tab-id>` the tabs should display using the ratatui tab's and the tab should show as `<tab-id>. <tab-name>`.

## Async editing

Async editing refers to never blocking user interaction. How we accomplish this is by leveraging a queue per chat thread. We will basically have a pending state for our chat history, so we will add messages (in sequence) from the history that need to be updated to a queue. So when we send additional messages we will wait till the previous message has been generated then send that message to our LLM. Using the spinners to show the message as not yet generated until our LLM starts populating them. We should take extra care to make sure user interaction, cursor, and prompts can handle the UI updating with the messages so the user can still type / navigate as the messages are coming through. All this to say we basically move to a queueing model per thread, we will leverage this in the next section so we should aim to make this have a nice API that can be re-used (for example take a window of messages and re-process them with different configuration).

## New message type, command (should be editable)

System messages currently display information and commands, instead we want to make it possible to display successful command results. If we insert on a command box we should open the command pallete (with the auto complete). We prepopulate the command from the history into the command pallete. However once we send a new valid command it behaves slightly different to editing a prompt message. When we edit a prompt message in the history we delete any messages that occured after it, when we change a command message we should actually start "replay" any message between two "model" commands with the model we've changed the command too. The same should occur between tools changes, so for example if we have these messages (denoted as prompt or cmd)
```
cmd: tools enable
prompt: tools are enabled
prompt: tools are fun!
cmd: tools disabled
```

and we edit the initial cmd to be `tools disabled` we look through messages after until we find another call of `tools` if it's the same command (like our `tools disabled` message we delete the second `tools disabled` from the history. if the message is not the same i.e `tools enabled` then we take all the messages between the two tools commands and queue them (in order synchronously) to be re-ran with the new settings set. If theres no command that mtches found or we've delete (compacted) the command we rerun all messages from the command we changed through the llm. We can view this as finding a window of messages that should be enqueued depending on tool calling and then using our new thread based queueing system to ensure they synchronously get executed one another and maintain the right chat history and configuration being sent to the LLM's.


## Break history

Add a new command, `break` break simply disconnects the history at that point. For example:
```
prompt: prompt 1
cmd: break
prompt: prompt 2
```

the chat log we send to the LLM should be just `prompt 2`.

## Deleting messages (new floating confirmation window)

When we press `d` in normal mode we shoud display a floating window like the cmd line in the same place, it should have a simple message `Confirmation: delete message [y/n]`. If the user pressed `n` we return to normal mode. If we press `y` we remove the message from the message log. For example we want to be able delete a break so we will now send the history that was broken.

```example
prompt: prompt 1
cmd: break
prompt: prompt 2
```
would become

```example
prompt: prompt 1
prompt: prompt 2
```

so now when we replay all the messages above that break.
