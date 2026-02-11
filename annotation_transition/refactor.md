Needs

input handler keyboard 
input handler mouse
input policy
normalize
draw rectangle and mask
show image
hold drawing state
hold rectangle and poly drawing
select label
render select label btns

Classes

OpencvRenderer -> Render tools


opencv input -> chama policy entregando intent -> policy executa o que acontece
para o callback é apenas um cmd = policy.evaluate(event, engine)


annotation overlay -> draw guidelines, render annotations
annotation view -> base render
keyboard and mouse handlers
command intent, draw state -> semantic representations
annotation mapper -> boundary mapper
command policy -> intent and state to action (semantic action)
annotation engine -> handle hard annotation logic
annotation pipeline -> hold annotations states, handle images order, save and load annotations (in other words, annotation session)
dataset_navigator
annotation_repository -> handle annotation storage

must change when use web:
keyboard and mouse handlers not used, keyboard must only set intent, to be replicate using web adapter
annotation view, annotation overlay not used
to do it, engine must only knows a UI adapter, another layer must build annotation pipeline, to can change adapter
between opencv renderer adapter and web comm adapter

In another words, pipeline must handle input and output as intents, even using DI

in this architecture, pipeline works as composition core, to build whole annotation system
Note that inference was not included yet