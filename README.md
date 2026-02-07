# Simplified-LLMs-Visualized
This project is a simplified, visual educational tool designed to peel back the curtain on Large Language Models.  
It was originally developed for DSAIY in collaboration with researchers from MIT, Brown, and Harvard to help pre-college students understand that AI is not magic. 

Note that this project is not designed to build chatbots, agents, or production-ready software. It is more of a scientific instrument designed to let you visualize how different AI models essentially think, predict, and behave under different conditions. There are several design decisions in this project that under normal conditions would be poor, such as the unloading of memory. However, these choices were made because most people using this project will have very limited familiarity with Python or coding. Requiring students to navigate files and change various settings would be challenging. This project was done in my limited spare time in a few days over a few hours, so I couldn't make the effort to optimize everything properly, especially for an educational context perfectly.

While I aimed for maximum accuracy, errors may still occur and or be present. Please feel free to reach out or submit a pull request with any corrections.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EvanFarnping/Simplified-How-LLMs-Work-Visualized/blob/main/Start_Here.ipynb)

## How Do LLMs Actually Work

Before running the code, it helps to understand what you are looking at.

### 1. It is Not Reading, It is Tokenizing

When you type a sentence into an AI, it does not see words the way you do. It breaks text down into small chunks called Tokens. A token can be a whole word, part of a word, or even a space.

The Visualization: You will see raw tokens in the charts. This is the raw language of the machine.

### 2. The Prediction Machine

At their core, LLMs are giant Next-Token Predictors. They do not know the answer. They calculate the statistical probability of what piece of text likely comes next based on everything they have ever read during training.

The Visualization: Our charts show you the Menu of Options the AI considered before picking an answer. You will see that sometimes the AI is 99 percent sure, and other times it is only 20 percent sure.

### 3. Temperature and Choice

If an AI always picked the number one most probable word, it would be boring and repetitive. Creativity in AI is often just the computer rolling a dice to pick the second or third best option.

## What This Tool Visualizes

This project contains several visualizations of different aspects of the AI brain. You can toggle these on or off in main_configs/main.py. Or if you want to try more specific experiments, see scenarios_to_try/{{EXPERIMENT}}.py

### 1. Prediction Charts

This creates a simple bar chart showing the top 10 tokens the model wanted to pick next.

What to look for: Does the model know the answer with high probability? Or is it guessing with low probability across many options?

### 2. Sequence Generation

This visualizes the model writing a sentence step-by-step.

What to look for: Watch how the probabilities change. If the model makes a mistake in Step 1, does it double down on the mistake in Step 2? This is often associated with "hallucination."

### 3. Comparison Video

This generates a video comparing how the model reacts to two different prompts side-by-side.

Example: Compare "Explain this simply" versus "Explain this like a scientist."

What to look for: See how a single word in your prompt changes the mathematical probabilities of the output.

### 4. The Sentiment Compass

We map the predicted words of the AI onto a psychological chart called the Circumplex Model of Affect. This is hard coded in the emotion_map_manager.py.

Y-Axis Activity: High Energy versus Low Energy.
X-Axis Valence: Positive Feeling versus Negative Feeling.

What to look for: If you act sad, does the AI move into the Sad or Passive quadrant?  
If you act aggressive, does it move to the Angry or Active quadrant?  
Note how the model considers certain tokens based on its Persona and given Prompt.  
Now imagine if the model had a very high temperature setting, will it be more likely to suggest a negative word now that may influence you as the user?

### 5. Brain Scan

This visualizes the internal Attention Layers of the model. Both broken down, and averaged.

What to look for: As the model reads a sentence, you can see which tokens it is focusing on. Is it paying attention to the subject? The verb? Or is it distracted? What do these patterns suggest?

## The Models

This project allows you to swap different AI brains. We use specific models to demonstrate how AI has evolved.

The Ancestors
GPT-2 and Pythia are older, smaller models. They struggle with logic but are great for seeing the raw basics of how LLMs learn grammar.

The Efficient Class
TinyLlama-1.1B and Phi-4-mini are modern small models. They are surprisingly smart despite fitting on a laptop. They show that architecture matters more than just size.

The Medium-Large Class
Mistral-7B and DeepSeek-Lite are capable models that rival the early versions of ChatGPT. They can reason, code, and chat fluently.

Take a look at model_manager.py to learn more about different models.

## Personas

You will see a file called personas.yaml. This controls the System Prompt. These are the hidden instructions given to the AI before you even start talking, kind of like how ChatGPT uses GPT under the hood, where ChatGPT has a special system prompt you can't see.

Examples:
Sad: Forces the model to select lower-energy words.
Caveman: Breaks the grammar capabilities of the model.
Liar: Intentionally aligns the model to output false information.
Direct: Strips away the polite Assistant personality.

This demonstrates that Personality in AI is just a set of hidden instructions steering the probabilities.

## How to Run It

1. Install Dependencies
Make sure you have Python installed, then run the pip install command for the requirements.txt file.
pip install -r requirements.txt
2. Configure Your Experiment
Open main_configs/main.py. This is your control center.
Change SELECTED_MODEL to swap LLMs.
Change CURRENT_PERSONA to change the personality.
Set RUN variables to True or False to choose which charts to generate.
3. Run the Engine
Run the main.py file using Python.
Check the export folder for your results.
4. Experiments
There are some pre-designed experiments to showcase fundamental LLM logic, feel free to simply click play and swap things out for whatever you like/can run.

## A Note on Hardware

LLMs are heavy, large, compuational intensive.

TinyLlama and GPT-2 will run on most modern laptops.
Mistral and DeepSeek require a computer with a good graphics card and decent RAM. If the program crashes, try a smaller model.
Phi-4-mini will likely run on any computer built after COVID-19 Era.

---

## How to Run It For Google Colab

The easiest way to use this project is via the included `Start_Here.ipynb` notebook.  
It acts as a control panel for the entire project.  

Make sure to download this project via https://github.com/EvanFarnping/Simplified-How-LLMs-Work-Visualized/blob/main/Start_Here.ipynb

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EvanFarnping/Simplified-How-LLMs-Work-Visualized/blob/main/Start_Here.ipynb)

### Step-by-Step Instructions
0.  **Open `Start_Here.ipynb`**: Launch the notebook in Google Colab (Jupyter Notebook should also work but I never tested this on that).
1.  **Run Step 0 (GPU Check)**
    * Click "Run" on the cell.
    * Check that you are using a GPU, as that is better. The bigger (VRAM) the better in this case.
2.  **Run Step 1 (Installation)**:
    * Click "Run" on the cell.
    * **CRITICAL:** This step installs required libraries and **will automatically crash/restart the runtime**.
    * *Ignore the "Session Crashed" popup.* This is necessary to apply the new libraries.
3.  **Run Step 1.5 (Admin Pre-Load) [Optional]**:
    * *Recommended for Classrooms.* Downloads the model weights to the disk cache immediately so students don't have to wait 5-10 minutes during the lesson.
    * Does NOT load models into RAM, so it is safe to run before class starts.
4.  **Run Step 2 (Load Engine)**:
    * Initializes the `main.py` engine and logs into Hugging Face.
    * If you skip this, nothing else will work.
5.  **Run Step 3 (Scenarios)**:
    * Select a pre-built experiment from the dropdown menu (e.g., `knowledge_cutoff`, `bias_in_roles`).
    * **Note:** Some scenarios require you to run them once, *edit the underlying file*, and run them again to see the difference.
6.  **Run Step 4 (Custom Lab Bench)**:
    * Use this to run your own custom tests without editing code files.
    * You can select any Model and Persona to test specific prompts.

---

## Troubleshooting & Edge Cases

### 1. The "Session Crashed" Popup (RAM Limit)
* **Symptom:** You try to load a model (e.g., `Mistral-7B`) and the session immediately restarts or goes black.
* **Cause:** You ran out of RAM (Random Access Memory). Standard Google Colab instances don't have a lot of RAM. Loading two models back-to-back or one massive model can cause a crash.
* **Solution:**
    * You do **NOT** need to reinstall libraries (Step 1).
    * Simply run **Step 2 (Load Engine)** again to re-initialize.
    * Choose a smaller model (e.g., `Phi-4-mini` or `TinyLlama`).

### 2. The "Disk Full" Error (Storage Limit)
* **Symptom:** You get an `OSError: [Errno 28] No space left on device` while downloading.
* **Cause:** You tried to download a model larger than the available disk space (usually ~70GB on free Colab).
    * *Example:* `Qwen2.5-72B` requires 140GB+ of space.
* **Solution:**
    * You must **Factory Reset** the runtime to clear the half-downloaded file.
    * Go to: `Runtime` -> `Disconnect and Delete Runtime`.
    * Restart from Step 1.
    * **Do not** attempt to run "Heavyweight" models on standard instances. 

### 3. The "Too Many Requests" Error (Hugging Face)
* **Symptom:** You are running things as usual and the terminal responds that you are sending too many requests. HuggingFace will likely be mentioned.
* **Cause:** The Token associated with HuggingFace is free, but does have rate limits.
    * *Example:* In Step 2, there is a token that is used to access the information from HuggingFace, the place with the AI models we are getting. 
* **Solution:**
    * You must switch out the **HuggingFace Token** in Step 2.
    * Delete the old Token and then uncomment the other options. You can also just create your own if you are comfortable doing so.
    * Restart from step 2 to load in the new Token to have access again.

### 4. "Module Not Found" Error
* **Cause:** You likely ran Step 2 before Step 1 finished, or after a factory reset.
* **Solution:** Run **Step 1**, wait for the crash/restart, then run **Step 2**.

---

## 🎓 For Teachers & Admins 🎓

### The "Admin Pre-Load" Workflow
If you are teaching a class, internet speeds can be a bottleneck.
1.  Open the notebook on the presentation computer (or student computers) **before class**.
2.  Run **Step 1**.
3.  Run **Step 1.5 (Admin Pre-Load)**. Select `Phi-4-mini`, `TinyLlama`, `Mistral`, and so on for whatever you want as an option.
4.  Wait for the downloads to finish.
5.  Run **Step 2** to initialize the engine and setup parameters + credentials.
6.  Wait for the downloads to finish.
7.  **Do not close the tab.**
8.  When class starts, students can run scenarios instantly because the 10-100GB+ of files are already cached on the disk.

### Editing Scenarios
Advanced students can edit the core logic files directly in the browser:
1.  Click the **Folder Icon (📁)** on the left sidebar.
2.  Navigate to `Simplified-How-LLMs-Work-Visualized/scenarios_to_try/`.
3.  Double-click any `.py` file to modify the experiment parameters (e.g., changing the prompt in `bias_in_roles/scenario.py`).
4.  Press `Ctrl+S` to save, then re-run the scenario cell.

---

## Recommended Hardware / Models in Google Colab (Ideally Pro):
### Order to Use:
1. H100 (Always try to use this if possible)
2. A100 
3. V6e-1 TPU 
4. L4 GPU
5. V5e-1 TPU
6. T4 GPU
7. CPU (Ideally never select this)

### Stats (You likely don't need to know this. Above info should suffice): 
CPU Only:
* N/A | ~13 GB (Std) / 51 GB (High) | vCompute: 2 – 8 | Baseline Performance

T4 GPU:	
* VRAM: 16 GB | System RAM: GDDR6 ~13 GB (Std) / 51 GB (High) | vCompute: 2 – 8 | Performance: 65 TFLOPS

L4 GPU:	
* VRAM: 24 GB GDDR6 | System RAM: ~53 – 64 GB | vCompute: 12 | Performance: 242 TFLOPS

A100 GPU:	
* VRAM: 40 GB / 80 GB* | System RAM: ~84 – 167 GB | vCompute: 12 | Performance: 312 TFLOPS

H100 GPU:	
* VRAM: 80 GB HBM3 | System RAM: ~84 – 90 GB+ | vCompute: 12 | Performance: 1,000+ TFLOPS

v5e-1 TPU:	
* VRAM: 16 GB HBM2 | System RAM: ~48 GB | vCompute: 24 | Performance: 197 TFLOPS

v6e-1 TPU:	
* VRAM: 32 GB HBM3 | System RAM: ~176 GB | vCompute: 44 | Performance: 918 TFLOPS

Created by **Evan Farnping** 2026
* **LinkedIn:** https://www.linkedin.com/in/evanfarnping/
* **Personal Website:** https://www.evanfarnping.com/

License: MIT License. Free for educational use.