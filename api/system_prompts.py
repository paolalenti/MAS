architect_prompt = open("system_prompts/architect.md", encoding="utf-8").read()
content_creator_prompt = open("system_prompts/content_creator.md", encoding="utf-8").read()
quiz_master_prompt = open("system_prompts/quiz_master.md", encoding="utf-8").read()


def get_prompt(base_prompt: str, **context) -> str:
    prompt = base_prompt
    for kwarg, value in context.items():
        prompt = prompt.replace(f"%{kwarg.upper()}%", value)
    return prompt
