from langchain.schema.messages import SystemMessage, HumanMessage

def build_messages(system_prompt: str, user_prompt: str):
    messages = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=user_prompt))
    return messages
