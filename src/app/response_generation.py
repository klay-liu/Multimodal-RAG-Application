# Creates the final response
from openai import OpenAI

def generate_rag_response(prompt, matched_items):
    
    # Create context
    text_context = ""
    image_context = []
    
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

    
    
    for item in matched_items:
        if 'text' in item.keys(): 
            text_context += str(item["page"]) + ". " + item['text'] + "\n"
        else:
            image_context.append(item['image'])
    
    final_prompt = f"""You are a helpful assistant for question answering.
    The text context is relevant information retrieved.
    The provided image(s) are relevant information retrieved.
    
    <context>
    {text_context}
    </context>
    
    Answer the following question using the relevant context and images.
    
    <question>
    {prompt}
    </question>
    
    Answer:"""
    if image_context:
        response = client.chat.completions.create(
            model="gemma-3-27b-it",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": final_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_context}"},
                        },
                    ],
                }
            ],
            max_tokens=500 # 根据需要调整
        )
    else:
        response = client.chat.completions.create(
            model="gemma-3-27b-it",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": final_prompt},
                    ],
                }
            ],
            max_tokens=500 # 根据需要调整
        )
    result = response.choices[0].message.content

    return result