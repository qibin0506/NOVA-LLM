# -*- coding: utf-8 -*-
# Keep existing imports (gevent, bottle, time, json, random)
from gevent import monkey; monkey.patch_all()
from bottle import Bottle, request, response, run, static_file, ServerAdapter
import time
import json
import random
from typing import Optional
from llm_model import LlmModel
from utils import init_env, get_model_config
import torch
from llm_trainer import streaming_generate

devices = "cpu"
if torch.cuda.is_available():
    devices = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    devices = "mps"

global_models = None

def llm_response(message: str, model: Optional[LlmModel]):
    """
    Simulates LLM response, now including Markdown and LaTeX examples.
    """
    answer_parts = ''
    full_parts = ''
    generator = streaming_generate(
        model=model,
        prompt=message,
        max_position_embeddings=1024,
        max_new_tokens=1024,
        temperature=0.9,
        p=0.75,
        device=devices
    )

    type = 'answer_chunk'
    for chunk in generator:
        full_parts = f'{full_parts}{chunk}'
        if chunk == '</s>': break
        if chunk == '</reasoning>' or chunk == '</answer>': continue
        if chunk == '<reasoning>':
            type = 'thinking_chunk'
            continue
        elif chunk == '<answer>':
            type = 'answer_chunk'
            continue

        answer_parts = f'{answer_parts}{chunk}'
        yield {"type": type, "content": chunk}

    yield {"type": "done", "content": answer_parts}
    print(f"Simulation finished. {full_parts}")


# --- Bottle app setup (keep existing routes: /, /chat) ---
app = Bottle()

@app.route('/')
def index():
    return static_file('/static/index.html', root='.')

@app.get('/static/<filename>')
def server_static(filename):
    return static_file(filename, root='./static')

@app.route('/chat', method='POST')
def chat_stream():
    model = None
    try:
        data = request.json
        user_message = data.get('message', '')
        model_name = data.get('model', 'dpo')
        model = global_models[model_name]
    except Exception as e:
        print(f"Error parsing request: {e}")
        response.status = 400
        return json.dumps({"error": "Invalid request format."})

    if not user_message:
        response.status = 400
        return json.dumps({"error": "Message cannot be empty."})

    response.content_type = 'text/event-stream'
    response.cache_control = 'no-cache'
    response.headers['Connection'] = 'keep-alive'

    def event_generator():
        try:
            for chunk in llm_response(user_message, model):
                yield f"data: {json.dumps(chunk)}\n\n"
        except Exception as e:
            print(f"Error during SSE generation: {e}")
            error_chunk = {"type": "error", "content": f"服务器处理出错: {e}"}
            yield f"data: {json.dumps(error_chunk)}\n\n"
        finally:
            print("SSE stream finished for this request.")

    return event_generator()

# --- Gevent Server Adapter (keep existing) ---
class GeventServerAdapter(ServerAdapter):
     def run(self, handler):
        from gevent.pywsgi import WSGIServer
        try:
            server = WSGIServer((self.host, self.port), handler, spawn='default')
            print(f"Gevent server starting on http://{self.host}:{self.port}")
            server.serve_forever()
        except KeyboardInterrupt:
            print("Server stopped by user.")
            server.stop()
        except Exception as e:
            print(f"Failed to start server: {e}")


def _init_model(model_name: str) -> LlmModel:
    model = LlmModel(get_model_config()).to(devices)
    model.load_state_dict(torch.load(f'modeling/{model_name}', weights_only=True))
    return model

if __name__ == '__main__':
    init_env()

    global_models = {
        'dpo': _init_model('dpo.bin'),
        'reasoning': _init_model('reasoning.bin'),
        'grpo': _init_model('grpo.bin')
    }

    run(app, host='localhost', port=8080, server=GeventServerAdapter, debug=True, reloader=True)