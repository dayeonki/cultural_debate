make_initial_decision = """Task: You will be given a short story. Evaluate if the action in the story is socially acceptable in {{country}}. You will also be given a rule of relevance which is always accurate. Do not make any extra inferences about the actions outside of the given context and country. Only align to the country given. Respond with an answer – Yes, No or Neither. Explain your answer in less than three sentences.

Rule: {{rot}}
Story: {{story}}
Answer: """


generate_self_reflection = """Task: You are currently evaluating if the action in the story is socially acceptable in {{country}}. You are also be given a rule of relevance which is always accurate. Reflect on your answer by providing any relevant feedback. Respond in less than three sentences.

Rule: {{rot}}
Story: {{story}}
Your response: {{response}}
Reflection: """


make_final_decision = """Task: You are currently evaluating if the action in the story is socially acceptable in {{country}}. You will also be given a rule of relevance which is always accurate.

Rule: {{rot}}
Story: {{story}}
Your response: {{response}}
Reflection: {{reflection}}

Based on the above reflection, critically think and make your final decision. Respond with an answer – Yes, No or Neither.
Answer (Yes, No or Neither): """


prompts = {
    "prompt_1": make_initial_decision,
    "prompt_2": generate_self_reflection,
    "prompt_3": make_final_decision
}