# References:
# https://huggingface.co/docs/transformers/en/model_doc/t5
# https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
# Refiner: https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/tree/main

from ImageEvaluator import Clip_Score_Eval
from transformers import T5ForConditionalGeneration, T5Tokenizer
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
import torch
from PIL import Image
import numpy as np
import os
from datasets import load_dataset, Dataset
from natsort import natsorted


if __name__ == '__main__':

    runName = "Run2"
    numImages = 100
    dataset = load_dataset("groloch/stable_diffusion_prompts_instruct")
    testSet = dataset["test"]

    if not os.path.exists("./Inference/"+str(runName)+"/Generatednp/"): 
        os.makedirs("./Inference/"+str(runName)+"/Generatednp/") 

    if not os.path.exists("./Inference/"+str(runName)+"/GeneratedImages/"): 
        os.makedirs("./Inference/"+str(runName)+"/GeneratedImages/") 

    model = T5ForConditionalGeneration.from_pretrained("./results/Run1/checkpoint-5755")
    tokenizer = T5Tokenizer.from_pretrained("./results/Run1/checkpoint-5755")

    # The following lines were used for manual prompt image generation/testing
    # userInput = input("Input original prompt: ")
    # userInput = "translate basic prompt to detailed longer prompt: " + userInput

    # First 5 prompts in the test set
    # originalPrompts = ["Natalie Portman as cheerful medieval innkeeper in a cozy, candlelit inn",
    #                    "Mystical goddess in tribal, techno-elven scene, 3D, cinematic.",
    #                    "Cyberpunk Woman in Neon City Sunset.",
    #                    "Cyborgs Sweaty Forehead, Hajime Sorayama Style",
    #                    "Max Headroom in Sci-Fi Perfume Ad, Large Eyes, Luxury Brands."
    #                    ]
    
    originalPrompts = np.random.choice(testSet, numImages, replace=False)
    originalPrompts = [list(d.values())[0] for d in originalPrompts]
    print(originalPrompts)
    generatedPrompts =[]

    for prompt in originalPrompts:
        input_ids = tokenizer("translate basic prompt to detailed longer prompt: " + prompt, return_tensors="pt").input_ids # DONT FORGET TO INCLUDE PREFIX
        outputs = model.generate(input_ids, max_length=1000, min_length=40,
                                    do_sample=True, repetition_penalty=1.4)
        generatedInput = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generatedPrompts.append(generatedInput)

    print("Original Prompts: " + str(originalPrompts))
    print("Generated Prompts: " + str(generatedPrompts))

    # Evaluator Class
    imageEvaluator = Clip_Score_Eval()

    # Generate both images
    # Note: Apparently StableDiffusionXLPipeline is different than StableDiffusionPipeline and I am using the XL Model
    sd_pipeline_base = StableDiffusionXLPipeline.from_single_file("./StableDiffusionModel/sd_xl_base_1.0.safetensors", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    sd_pipeline_base.to("cuda")

    sd_pipeline_refiner = StableDiffusionXLImg2ImgPipeline.from_single_file("./StableDiffusionModel/sd_xl_refiner_1.0.safetensors", text_encoder_2=sd_pipeline_base.text_encoder_2, vae=sd_pipeline_base.vae, torch_dtype=torch.float16, use_safetensors=True,variant='fp16',)
    sd_pipeline_refiner.to("cuda")

    # Steps for refiner
    n_steps = 40
    high_noise_frac = 0.8

    numImagesPerPrompt = 1

    combinedPrompts = originalPrompts + generatedPrompts

    for i, prompt in enumerate(combinedPrompts):
        image = sd_pipeline_base(prompt=prompt, num_inference_steps=n_steps, denoising_end=high_noise_frac, output_type='latent', height=512, width=512).images
        image = sd_pipeline_refiner(prompt=prompt, num_inference_steps=n_steps, denoising_start=high_noise_frac,image=image, output_type='np').images[0]
        np.save(f'./Inference/{runName}/Generatednp/Image-{i}.npy',image, allow_pickle=True)

    combinedOutput = []
    for file in natsorted(os.listdir("./Inference/"+runName+"/Generatednp")):    # This is a really weird way for me attempting to get around memory issues by generating np arrays of the images and saving them
        imageArrayLoaded = np.load("./Inference/"+runName+"/Generatednp/" + file, allow_pickle=True)
        combinedOutput.append(imageArrayLoaded)

    combinedOutput = np.array(combinedOutput)
    userInputImages = combinedOutput[:numImages]
    generatedInputImages = combinedOutput[numImages:]

    for i, image in enumerate(userInputImages):
        saveUserInputImage = Image.fromarray((image*255).astype(np.uint8))  # must convert to 0-255 range (not float)
        saveUserInputImage.save(f"./Inference/{runName}/GeneratedImages/UserInputImage-{i}.jpeg")

    for i, image in enumerate(generatedInputImages):
        saveGeneratedInputImage = Image.fromarray((image*255).astype(np.uint8))
        saveGeneratedInputImage.save(f"./Inference/{runName}/GeneratedImages/GeneratedInputImage-{i}.jpeg")

    print(f"Original Prompt CLIP Score: {imageEvaluator.calculate_clip_score(userInputImages, originalPrompts)}")
    print(f"Generated Prompt CLIP Score (Based on New Prompt/New Image): {imageEvaluator.calculate_clip_score(generatedInputImages, generatedPrompts)}")
    print(f"Generated Prompt CLIP Score (Based on Original Prompt/New Image): {imageEvaluator.calculate_clip_score(generatedInputImages, originalPrompts)}")