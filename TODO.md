# Features

# Long response scroll behaviour

When we have a long response we currently can't scroll and see more of the message, we return blocks of parsed markdown and tool calls from our agent, to fix this we're going to need to add to vim's modes. We'll have a `explore` mode entered with `e` for now this will just allow us to use hjkl vim motions to navigate through the text, which will scroll appropriately when we reach the bottom of our explorable space.
