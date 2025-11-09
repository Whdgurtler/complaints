import gradio as gr
from transformers import pipeline
import torch

# Initialize a conversational pipeline with a model (e.g., "facebook/blenderbot-400M-distill")
# Note: Larger models may require significant VRAM/compute
try:
    chatbot_pipeline = pipeline(model="facebook/blenderbot-400M-distill", task="conversational", device=0 if torch.cuda.is_available() else -1)
except Exception as e:
    print(f"Loading model failed, falling back to a dummy function: {e}")
    chatbot_pipeline = None

def llm_response(message, history):
    # Format the history into a format the pipeline expects (list of lists: [[user, bot], [user, bot], ...])
    conversation_history = []
    for user_msg, bot_msg in history:
        conversation_history.append({"role": "user", "content": user_msg})
        conversation_history.append({"role": "assistant", "content": bot_msg})
    
    # Add current user message
    conversation_history.append({"role": "user", "content": message})

    if chatbot_pipeline:
        # Generate response using the pipeline
        response = chatbot_pipeline(conversation_history)
        return response[-1]["content"] # Return only the latest bot response
    else:
        return "Model not loaded. Check your setup or use a dummy function."

# Create the Gradio ChatInterface
demo = gr.ChatInterface(
    fn=llm_response,
    title="Hugging Face LLM Chatbot",
    description="Chat with a language model hosted via Hugging Face and Gradio.",
    examples=["Hello!", "What can you do?", "How does this work?"],
    theme="soft"
)

if __name__ == "__main__":
    demo.launch()
