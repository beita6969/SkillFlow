


WEBSHOP_TEMPLATE_NO_HIS = """You are an expert autonomous agent operating in the WebShop e-commerce environment.
Your task is to: {task_description}.
Your current observation is: {current_observation}.
Your admissible actions of the current situation are:
[
{available_actions}
].

Now it's your turn to take one action for the current step.
Return exactly one executable action string in the form search[keywords] or click[value].
For click actions, copy one value from the admissible action list exactly. Do not repeat these instructions.
"""

WEBSHOP_TEMPLATE = """You are an expert autonomous agent operating in the WebShop e-commerce environment.
Your task is to: {task_description}.
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}.
Your admissible actions of the current situation are:
[
{available_actions}
].

Return exactly one executable action string in the form search[keywords] or click[value].
For click actions, copy one value from the admissible action list exactly. Do not repeat these instructions.
"""


WEBSHOP_TEMPLATE_NO_HIS_SKILLRL = """
You are an expert autonomous agent operating in the WebShop e-commerce environment. 
Your task is to: {task_description}.
Your current observation is: {current_observation}.
Your admissible actions of the current situation are: 
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

WEBSHOP_TEMPLATE_SKILLRL = """
You are an expert autonomous agent operating in the WebShop e-commerce environment.
Your task is to: {task_description}.
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}.
Your admissible actions of the current situation are:
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

WEBSHOP_TEMPLATE_WITH_MEMORY_SKILLRL = """
You are an expert autonomous agent operating in the WebShop e-commerce environment.
Your task is to: {task_description}.

## Retrieved Relevant Experience

{retrieved_memories}

## Current Progress

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}.
Your admissible actions of the current situation are:
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""


WEBSHOP_AGENTBENCH_SYSTEM = """
You are web shopping.
I will give you instructions about what to do.
You have to follow the instructions.
Every round I will give you an observation and a list of available actions,
you have to respond with one executable action based on the state and instruction.
You can use search if search is available.
You can click one of the buttons in clickables.
If the action is not valid, perform nothing.
Keywords in search are up to you, but the value in click must be a value in the list of available actions.
Remember that your keywords in search should be carefully designed.
You should first think about what to do, then output the corresponding action.
Output exactly one action string in the form search[keywords] or click[value].
"""

WEBSHOP_TEMPLATE_NO_HIS_AGENTBENCH = WEBSHOP_AGENTBENCH_SYSTEM + """

The initial observation:
{current_observation}

Available Actions:
{available_actions_dict}
"""

WEBSHOP_TEMPLATE_AGENTBENCH = WEBSHOP_AGENTBENCH_SYSTEM + """

Recent interaction history:
{action_history}

Observation:
{current_observation}

Available Actions:
{available_actions_dict}
"""


_ALFWORLD_EXAMPLE = """Action format examples:
> go to cabinet 1
> take apple 1 from countertop 1
> open fridge 1
> move apple 1 to fridge 1
> heat apple 1 with microwave 1
> clean mug 1 with sinkbasin 1
> cool potato 1 with fridge 1
> move plate 1 to countertop 1
> examine shelf 1
"""

ALFWORLD_TEMPLATE_NO_HIS = """You are an expert agent operating in the ALFRED Embodied Environment.
{example}
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action. Pick exactly one action from the admissible actions list.
Output ONLY the action you choose. No explanation, no reasoning, just the action.
"""

ALFWORLD_TEMPLATE = """You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
{example}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action. Pick exactly one action from the admissible actions list.
Output ONLY the action you choose. No explanation, no reasoning, just the action.
"""
