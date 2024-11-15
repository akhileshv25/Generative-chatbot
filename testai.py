import pandas as pd
from transformers import GPT2Tokenizer, GPTNeoForCausalLM
import torch
import re

model_name = "EleutherAI/gpt-neo-125M"  
model = GPTNeoForCausalLM.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

csv_file_path = 'lights.csv'  
light_data = pd.read_csv(csv_file_path)

def query_csv(question):
  
    question_lower = question.lower()

    if "status" in question_lower:
        intent = "status"
    elif "how many" in question_lower and "lights" in question_lower:
        intent = "count"
    elif "location" in question_lower:
        intent = "location"
    else:
        return "I'm sorry, I couldn't understand your question. Could you rephrase?"

    locations = light_data['location'].str.lower().unique()
    location_match = next((loc for loc in locations if re.search(rf"\b{loc}\b", question_lower)), None)

    if intent == "status" and location_match:
        filtered_lights = light_data[light_data['location'].str.lower() == location_match]
        if filtered_lights.empty:
            return f"No lights found in {location_match}."
        
        response = []
        for _, row in filtered_lights.iterrows():
            response.append(f"Light {row['id']} is {row['status']}.")
        return " ".join(response)

    elif intent == "count":
        count = len(light_data)
        return f"There are {count} lights in total."

    elif intent == "location":
        unique_locations = light_data['location'].unique()
        return f"The lights are located in the following areas: {', '.join(unique_locations)}."

    else:
        return "Please specify a location for status inquiries."

def generate_response(question, structured_answer):
    
    prompt = f"{question}\n{structured_answer}"
    
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs['input_ids'], max_length=150, do_sample=True, temperature=0.7)
    generated_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_answer

print("Welcome to the Light Status Assistant! Type 'exit' to end the conversation.\n")

while True:
    question = input("You: ")
    if question.lower() == "exit":
        print("Goodbye!")
        break

    structured_answer = query_csv(question)
    final_response = generate_response(question, structured_answer)
    
    print(f"Assistant: {final_response}\n")
