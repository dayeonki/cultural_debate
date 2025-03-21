prompts = {
    "en": """Task: You will be given a short story. Evaluate if the action in the story is socially acceptable in {{country}}. Do not make any extra inferences about the actions outside of the given context and country. Only align to the country given. Respond with an answer – Yes, No or Neither.

Story: {{story}}
Answer (Yes, No or Neither): """,

    "en_rot": """Task: You will be given a short story. Evaluate if the action in the story is socially acceptable in {{country}}. You will also be given a rule of relevance which is always accurate. Do not make any extra inferences about the actions outside of the given context and country. Only align to the country given. Respond with an answer – Yes, No or Neither.

Rule: {{rot}}
Story: {{story}}
Answer (Yes, No or Neither): """
}