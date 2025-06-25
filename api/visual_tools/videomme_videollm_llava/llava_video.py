from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Optional
import torch
import numpy as np
import warnings
from decord import VideoReader, cpu
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
import copy
import uvicorn

warnings.filterwarnings("ignore")

app = FastAPI()

# ---- Model Load ----
pretrained = "lmms-lab/LLaVA-Video-7B-Qwen2"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"

tokenizer, model, image_processor, max_length = load_pretrained_model(
    pretrained, None, model_name, torch_dtype="float16", device_map=device_map
)
model.eval()

# ---- Request Schema ----
class InferenceRequest(BaseModel):
    video_path: str
    prompt: Optional[str] = "Please describe this video in detail."
    max_frames: int = 64
    frame_step: int = 1

# ---- Video Loader ----
def load_video(video_path, max_frames_num, fps=1, force_sample=False):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps() / fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i / fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i / vr.get_avg_fps() for i in frame_idx]
    frame_time_str = ",".join([f"{i:.2f}s" for i in frame_time])
    frames = vr.get_batch(frame_idx).asnumpy()
    return frames, frame_time_str, video_time

# ---- Inference Endpoint ----
@app.post("/video/infer")
def infer_video(request: InferenceRequest):
    try:
        video_np, frame_time, video_time = load_video(
            request.video_path,
            request.max_frames,
            request.frame_step,
            force_sample=True
        )
        video_tensor = image_processor.preprocess(video_np, return_tensors="pt")["pixel_values"].cuda().half()
        video_tensor = [video_tensor]

        conv_template = "qwen_1_5"
        time_instruction = f"The video lasts for {video_time:.2f} seconds, and {len(video_tensor[0])} frames are uniformly sampled from it. These frames are located at {frame_time}. {request.prompt}"

        question = DEFAULT_IMAGE_TOKEN + "\n" + time_instruction
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                images=video_tensor,
                modalities=["video"],
                do_sample=True,
                temperature=0.7,
                max_new_tokens=4096,
            )

        response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return {"response": response}
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            torch.cuda.empty_cache()
            return {"error": "CUDA out of memory. Try reducing max_frames or video resolution."}
        else:
            raise e
    except Exception as e:
        return {"error": str(e)}
if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    uvicorn.run("llava_video:app", host="0.0.0.0", port=8000)