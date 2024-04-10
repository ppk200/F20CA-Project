import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from threading import Thread
from conversation import get_default_conv_template



#tokenizer = AutoTokenizer.from_pretrained("GeneZC/MiniChat-3B")
#model = AutoModelForCausalLM.from_pretrained("GeneZC/MiniChat-3B", torch_dtype=torch.float16)

print(torch.cuda.is_available())

tokenizer = AutoTokenizer.from_pretrained("GeneZC/MiniChat-3B", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("GeneZC/MiniChat-3B", use_cache=True, torch_dtype=torch.float16,device_map={'': 0})

model = model.to('cuda:0')

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [29,0] # This isn't right (not sure what to put here)
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id: #idk about this too
                return True
        return False



def predict(message, history):
    if(not history):
        history = []
    
    history_transformer_format = history + [[message, ""]]
    stop = StopOnTokens()
    conv = get_default_conv_template("minichat")


    messages = "".join(["".join(["\n<human>:"+item[0], "\n<bot>:"+item[1]])  #curr_system_message +
                for item in history_transformer_format])
        
    conv.append_message(conv.roles[0], messages)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer([prompt], return_tensors="pt").to("cuda")

    streamer = TextIteratorStreamer(tokenizer, timeout=30., skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        input_ids,
        streamer=streamer,
        max_new_tokens=1024,
        do_sample=True,
        #top_p=0.95,
        #top_k=1000,
        temperature=0.7,
        #num_beams=1,
        stopping_criteria=StoppingCriteriaList([stop])
        )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    partial_message  = ""
    for new_token in streamer:
        if new_token != '<':
            partial_message += new_token
            #yield partial_message , history

    if(partial_message[:6] == "<bot>:"):
        history.append([message, partial_message[6:]])
    else:
        history.append([message, partial_message])

    #print (history)
    return history , history


demo = gr.Interface(fn=predict,
                    allow_flagging="manual",
                    title = "Chatbot C",
                    description = "You're currently interacting with Chatbot C",
                    inputs=["text", gr.State()],
                    theme="monochrome",
                    outputs=["chatbot", gr.State()])
demo.launch()

"""
demo = gr.Interface(fn=predict,
                    inputs=["text", gr.State()],
                    outputs=["text", gr.State()])


demo.launch()
"""

#gr.ChatInterface(predict).launch()