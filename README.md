# CAP6640 Semester Project
## Please see the included report for a comprehensive overview
## Description
-SemesterProject.py: This script is where the training of the FLAN-T5 model takes place. Run with "python3 .\SemesterProject.py". Note: you must obtain the Stable Diffusion base model from: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0.safetensors as well as the refiner from: https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/blob/main/sd_xl_refiner_1.0.safetensors. After downloading put them into the ./CAP6640SemesterProject/StableDiffusionModel folder. After training the top 5 best models will be saved to "./CAP6640SemesterProject/results"

-Inference.py: This script generates the improved prompt pairs. Run with "python3 .\Inference.py". Modify the parameters at the top of the script such as "runName" or "numImages" to change where the run is saved and how many images/prompts to generate. Generated images will save in "./CAP6640SemesterProject/Inference/{runName}/GeneratedImages"

-ImageEvaluator.py: This script can not be run directly and is just a helper function in order to calculate the CLIP score of the generated images compared to the prompts.

-LogHistoryVisualizer.py: This script generates graphs for visualizing loss, bertscore, etc. It is run by doing "python3 .\LogHistoryVisualizer.py"