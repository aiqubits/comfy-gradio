# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import gc
import os.path as osp
import os
import sys
import warnings
import json
import time
import random
import uuid

import gradio as gr
# import numpy as np
import requests
from prompt_extend import DashScopePromptExpander
from PIL import Image
from urllib.parse import quote

i2v_host = os.getenv('I2V_HOST', 'localhost')
i2v_port = os.getenv('I2V_PORT', '8188')
I2V_URL = f'http://{i2v_host}:{i2v_port}'
print(f"I2V_URL: {I2V_URL}")
GR_BASE_URL = "http://192.168.100.100:7860/"
current_directory = os.getcwd()
INPUT_DIR = os.path.join(current_directory, "input")
OUTPUT_DIR = os.path.join(current_directory, "output")
# CANCEL_FILE = os.path.join(OUTPUT_DIR, "_cancel_")
WORKFLOW_API = os.path.join(current_directory, "comfy-gradio", "i2v_workflow_api.json")


warnings.filterwarnings('ignore')

# Model
sys.path.insert(0, os.path.sep.join(osp.realpath(__file__).split(os.path.sep)[:-2]))

# Global Var
prompt_expander = None
wan_i2v_480P = None
wan_i2v_720P = None

interrupt_flag = False
unfreeze = True

def start_queue(prompt_workflow):
    p = {"prompt": prompt_workflow}
    data = json.dumps(p).encode('utf-8')
    requests.post(os.path.join(I2V_URL, "prompt"), data=data)

# Button Func
def load_model(value):
    global wan_i2v_480P, wan_i2v_720P

    if value == '------':
        print("No model loaded")
        return '------'

    if value == '720P':
        if args.ckpt_dir_720p is None:
            print("Please specify the checkpoint directory for 720P model")
            return '------'
        if wan_i2v_720P is not None:
            pass
        else:
            del wan_i2v_480P
            gc.collect()
            wan_i2v_480P = None

            print("load 14B-720P i2v model...", end='', flush=True)
            # cfg = WAN_CONFIGS['i2v-14B']
            cfg = ""
            # wan_i2v_720P = wan.WanI2V(
            #     config=cfg,
            #     checkpoint_dir=args.ckpt_dir_720p,
            #     device_id=0,
            #     rank=0,
            #     t5_fsdp=False,
            #     dit_fsdp=False,
            #     use_usp=False,
            # )

            print("done", flush=True)
            return '720P'

    if value == '480P':
        if args.ckpt_dir_480p is None:
            print("Please specify the checkpoint directory for 480P model")
            return '------'
        if wan_i2v_480P is not None:
            pass
        else:
            del wan_i2v_720P
            gc.collect()
            wan_i2v_720P = None

            print("load 14B-480P i2v model...", end='', flush=True)
            # cfg = WAN_CONFIGS['i2v-14B']
            cfg = ""
            # wan_i2v_480P = wan.WanI2V(
            #     config=cfg,
            #     checkpoint_dir=args.ckpt_dir_480p,
            #     device_id=0,
            #     rank=0,
            #     t5_fsdp=False,
            #     dit_fsdp=False,
            #     use_usp=False,
            # )

            print("done", flush=True)
            return '480P'


def prompt_enc(prompt, img, tar_lang):
    if (not unfreeze):
        print("prompt_enc no action")
        return prompt
    print('prompt extend...')
    # Image.Image
    if img is None:
        print('Please upload an image')
        return prompt
    print(f"prompt: {prompt}")
    if prompt == "":
        print('Please enter prompt')
        return prompt
    
    global prompt_expander
    prompt_output = prompt_expander(
        prompt, image=img, tar_lang=tar_lang.lower())
    if prompt_output.status == False:
        return prompt
    else:
        return prompt_output.prompt
    
def get_latest_image(folder):
    files = os.listdir(folder)
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort(key=lambda x: os.path.getmtime(os.path.join(folder, x)))
    latest_image = os.path.join(folder, image_files[-1]) if image_files else None
    return latest_image

def crop_image(input_image):
        # 直接处理 PIL.Image
    if isinstance(input_image, Image.Image):
        image = input_image
    else:
        # 如果输入是 numpy 数组，转换回 PIL.Image
        image = Image.fromarray(input_image)
    min_side = min(image.size)
    scale_factor = 512 / min_side
    new_size = (round(image.size[0] * scale_factor), round(image.size[1] * scale_factor))
    resized_image = image.resize(new_size)

    resized_image.save(os.path.join(INPUT_DIR, "tmp_input.jpg"))

def get_latest_video():
    files = os.listdir(OUTPUT_DIR)
    video_list = [f for f in files if f.lower().endswith(('.webp', '.mp4', '.mov', '_cancel_'))]
    video_list.sort(key=lambda x: os.path.getmtime(os.path.join(OUTPUT_DIR, x)))
    latest_video = os.path.join(OUTPUT_DIR, video_list[-1]) if video_list else None
    print(f"latest_video: {latest_video}")
    return [latest_video]

def get_video_list():
    files = os.listdir(OUTPUT_DIR)
    video_list = [f for f in files if f.lower().endswith(('.mp4', '.mov', '.webp'))]
    video_list.sort(key=lambda x: os.path.getmtime(os.path.join(OUTPUT_DIR, x)), reverse=True)
    video_list = video_list[:20]
    new_video_list = []
    for video in video_list:
        current_video = os.path.join(OUTPUT_DIR, video) if video else None
        new_video_list.append(current_video)

    print(f"video_list: {new_video_list}")
    return new_video_list

def get_video_list_wrapped():
    if (not unfreeze):
        print("get_video_list_wrapped no action")
        return get_latest_video()
    video_list = get_video_list()
    return video_list

def remove_video_list():
    if (not unfreeze):
        print("remove_video_list no action")
        return
    video_list = get_video_list()
    for video in video_list:
        os.remove(os.path.join(OUTPUT_DIR, video))

def i2v_generation(img2vid_prompt, img2vid_image, resolution, dimension, duration, index, n_prompt):
    global unfreeze
    unfreeze = False
    print(f"i2v_generation unfreeze: {unfreeze}")
    print("done", flush=True)
    print(f"{img2vid_prompt},{resolution},{duration},{index},{n_prompt}")
    if resolution not in ["480P", "720P"]:
        print(
            'Please specify the resolution'
        )
        return None
    
    if dimension not in ['512*512', '1280*720', '720*1280', '1080*1080']:
        print(
            'Please specify the dimension'
        )
        return None

    if img2vid_image is None or img2vid_prompt == "":
        print('Please upload an image and enter a prompt')
        return None
    
    else:
        crop_image(img2vid_image)
        with open(WORKFLOW_API, "r") as file_json:
            prompt = json.load(file_json)        
        # 请求comfyui api生成  img2vid_prompt
        client_id = str(uuid.uuid4())
        prompt["client_id"] = client_id
        
        # random seed
        if index == -1:
            # 1500000
            prompt["prompt"]["3"]["inputs"]["seed"] = random.randint(1, 1500000)

        # prompt
        prompt["prompt"]["6"]["inputs"]["text"] = img2vid_prompt

        # neginative
        if len(n_prompt) != 0:
            prompt["prompt"]["7"]["inputs"]["text"] = n_prompt

        # time duration
        prompt["prompt"]["50"]["inputs"]["length"] = duration * 16 + 1

        # dimension
        match dimension:
            case '512*512':
                prompt["prompt"]["50"]["inputs"]["width"] = 512
                prompt["prompt"]["50"]["inputs"]["height"] = 512
            case '1280*720':
                prompt["prompt"]["50"]["inputs"]["width"] = 1280
                prompt["prompt"]["50"]["inputs"]["height"] = 720
            case '720*1280':
                prompt["prompt"]["50"]["inputs"]["width"] = 720
                prompt["prompt"]["50"]["inputs"]["height"] = 1280
            case '1080*1080':
                prompt["prompt"]["50"]["inputs"]["width"] = 1080
                prompt["prompt"]["50"]["inputs"]["height"] = 1080

        prompt["prompt"]["52"]["inputs"]["image"] = os.path.join(INPUT_DIR, "tmp_input.jpg")
        load_image_str = {
					"id": 52,
					"type": "LoadImage",
					"pos": [20, 760],
					"size": [315, 314],
					"flags": {},
					"order": 2,
					"mode": 0,
					"inputs": [],
					"outputs": [{
						"name": "IMAGE",
						"type": "IMAGE",
						"links": [106, 109],
						"slot_index": 0
					},
					{
						"name": "MASK",
						"type": "MASK",
						"links": "None",
						"slot_index": 1
					}],
					"properties": {
						"Node name for S&R": "LoadImage"
					},
					"widgets_values": [os.path.join(INPUT_DIR, "tmp_input.jpg"), "image"]
				}
        prompt["extra_data"]["extra_pnginfo"]["workflow"]["nodes"].append(load_image_str)

        load_unet_480_str = {
					"id": 37,
					"type": "UNETLoader",
					"pos": [20, 70],
					"size": [346.7470703125, 82],
					"flags": {},
					"order": 4,
					"mode": 0,
					"inputs": [],
					"outputs": [{
						"name": "MODEL",
						"type": "MODEL",
						"links": [110],
						"slot_index": 0
					}],
					"properties": {
						"Node name for S&R": "UNETLoader"
					},
					"widgets_values": ["wan2.1_i2v_480p_14B_bf16.safetensors", "default"]
				}
        
        load_unet_720_str = {
					"id": 37,
					"type": "UNETLoader",
					"pos": [20, 70],
					"size": [346.7470703125, 82],
					"flags": {},
					"order": 4,
					"mode": 0,
					"inputs": [],
					"outputs": [{
						"name": "MODEL",
						"type": "MODEL",
						"links": [110],
						"slot_index": 0
					}],
					"properties": {
						"Node name for S&R": "UNETLoader"
					},
					"widgets_values": ["wan2.1_i2v_720p_14B_bf16.safetensors", "default"]
				}

        if resolution == "480p":
            prompt["extra_data"]["extra_pnginfo"]["workflow"]["nodes"].append(load_unet_480_str)
        else:        
            prompt["prompt"]["37"]["inputs"]["unet_name"] = "wan2.1_i2v_720p_14B_bf16.safetensors"
            prompt["extra_data"]["extra_pnginfo"]["workflow"]["nodes"].append(load_unet_720_str)

        img2vid_prompt_data = json.dumps(prompt).encode('utf-8')

        requests.post(os.path.join(I2V_URL, "prompt"), data=img2vid_prompt_data)

        previous_video = get_latest_video()
        print(f"previous_video: {previous_video}")
        # if (previous_video is not None) and ("_cancel_" in  previous_video):
        #     # os.remove(CANCEL_FILE)
        #     print(f"_cancel_ previous_video: {get_latest_video()}")

        while True:
            latest_video = get_latest_video()
            print(f"while True latest_video: {latest_video}")
            print(f"while True previous_video: {previous_video}")
            global interrupt_flag
            if interrupt_flag or (latest_video != previous_video):
                print(f"while True break flag: {interrupt_flag}")
                print("Generation over")
                break
            time.sleep(5)

        # if (latest_video is not None) and ("_cancel_" in latest_video):
        #     # os.remove(CANCEL_FILE)
        #     print("Generation canceled")
        #     return
        
        unfreeze = True
        print(f"unfreeze: {unfreeze}")
        print("i2v_generation done", flush=True)
        return latest_video

def interrupt_i2v_generation():
    # with open(CANCEL_FILE, 'w') as file:
        #pass  # Do nothing, just create the file    
    requests.post(os.path.join(I2V_URL, "interrupt"))
    global interrupt_flag
    interrupt_flag = True
    print("interrupt_i2v_generation flag: ", interrupt_flag)
    print("interrupt_i2v_generation done", flush=True)

def get_task_number():
    info_json = requests.get(os.path.join(I2V_URL, "prompt"))
    info_json = json.loads(info_json.text)
    print(f"info_json: {info_json}")
    task_number = info_json["exec_info"]["queue_remaining"]
    print(f"task_number: {task_number}")
    return task_number

# Interface
def gradio_interface():
    with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
        gr.Markdown("""
                    <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
                        办公智能化—AI视频生成云平台 (I2V-14B)
                    </div>
                    <div style="text-align: center; font-size: 16px; font-weight: normal; margin-bottom: 20px;">
                        中煤宣传部定制版
                    </div>
                    """)
        with gr.Tab("首页"):
            with gr.Row(variant="compact", show_progress=True):
                with gr.Column():
                    with gr.Row():
                        resolution = gr.Dropdown(
                            label='分辨率',
                            choices=['480P', '720P'],
                            value='480P')
                        dimension = gr.Dropdown(
                            label='尺寸',
                            choices=['512*512', '1280*720', '720*1280', '1080*1080'],
                            value='512*512')

                    img2vid_image = gr.Image(
                        type="pil",
                        label="上传输入图像",
                        elem_id="image_upload",
                    )
                    img2vid_prompt = gr.Textbox(
                        label="提示词",
                        placeholder="描述你想要生成的视频内容",
                    )
                    tar_lang = gr.Radio(
                        choices=["CH", "EN"],
                        label="提示词的语言",
                        value="CH")
                    run_p_button = gr.Button(value="优化提示词")

                    with gr.Accordion("高级选项", open=False):
                        with gr.Row():
                            duration = gr.Slider(
                                label="视频时长/秒",
                                minimum=1,
                                maximum=10000,
                                value=2,
                                step=1)
                            index = gr.Slider(
                                label="视频编号",
                                minimum=-1,
                                maximum=2147483647,
                                step=1,
                                value=-1)
                        # with gr.Row():
                        #     shift_scale = gr.Slider(
                        #         label="Shift scale",
                        #         minimum=0,
                        #         maximum=10,
                        #         value=5.0,
                        #         step=1)
                        #     guide_scale = gr.Slider(
                        #         label="Guide scale",
                        #         minimum=0,
                        #         maximum=20,
                        #         value=5.0,
                        #         step=1)
                        n_prompt = gr.Textbox(
                            label="负向提示词",
                            placeholder="描述你要添加的负向提示词"
                        )

                    run_i2v_button = gr.Button("生成视频")
                with gr.Column():
                    result_gallery = gr.Gallery(type="filepath", label='生成的视频', height=700, rows=2, columns=5)
                    check_latest_video = gr.Button("获取云端最近的20个视频")

                    delete_video_list = gr.Button("删除云端最近的20个视频(危险！删除后无法恢复)")
        with gr.Tab("任务管理"):
            global unfreeze
            with gr.Row(variant="compact", show_progress=True, visible=unfreeze):
                with gr.Column():
                    check_current_task = gr.Button("查看正在进行的视频生成任务个数")
                    run_interrupt = gr.Button("取消正在进行的视频生成任务")
                with gr.Column():
                    task_status = gr.Textbox(label="正在进行的视频生成任务个数")
                
        resolution.input(
            fn=None, inputs=[], outputs=[])
        
        dimension.input(
            fn=None, inputs=[], outputs=[])
        
        run_p_button.click(
            fn=prompt_enc,
            inputs=[img2vid_prompt, img2vid_image, tar_lang],
            outputs=[img2vid_prompt],
            show_progress="minimal",
        )

        run_i2v_button.click(
            fn=i2v_generation,
            inputs=[
                img2vid_prompt, img2vid_image, resolution, dimension, duration,
                index, n_prompt
            ],
            outputs=result_gallery,
            show_progress="full",
        )
        
        check_latest_video.click(
            fn=get_video_list_wrapped,
            inputs=[],
            outputs=result_gallery,
        )

        check_current_task.click(
            fn=get_task_number,
            inputs=[],
            outputs=[task_status],
            show_progress="full",
        )

        delete_video_list.click(
            fn=remove_video_list,
            inputs=[],
            outputs=[],
            show_progress="full",
        )
        
        run_interrupt.click(
            fn=interrupt_i2v_generation,
            inputs=[],
            outputs=[],
        )

    return demo


# Main
def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a video from a text prompt or image using Gradio")
    parser.add_argument(
        "--ckpt_dir_720p",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--ckpt_dir_480p",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--prompt_extend_method",
        type=str,
        default="local_qwen",
        choices=["dashscope", "local_qwen"],
        help="The prompt extend method to use.")
    parser.add_argument(
        "--prompt_extend_model",
        type=str,
        default=None,
        help="The prompt extend model to use.")

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = _parse_args()

    print("Step1: Init prompt_expander...", end='', flush=True)
    if args.prompt_extend_method == "dashscope":
        prompt_expander = DashScopePromptExpander(
            model_name=args.prompt_extend_model, is_vl=True)
    elif args.prompt_extend_method == "local_qwen":
        prompt_expander = DashScopePromptExpander(
            model_name=args.prompt_extend_model, is_vl=True, device=0)
    else:
        raise NotImplementedError(
            f"Unsupport prompt_extend_method: {args.prompt_extend_method}")

    # if CANCEL_FILE in os.listdir(OUTPUT_DIR):
    #     os.remove(CANCEL_FILE)
    
    print("done", flush=True)
    demo = gradio_interface()
    demo.launch(server_name="0.0.0.0")
