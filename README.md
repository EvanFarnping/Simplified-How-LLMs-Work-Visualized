# Simplified-LLMs-Visualized (OUTDATED, NEEDS UPDATE)
This project is a simplified, visual educational tool designed to peel back the curtain on Large Language Models. It was originally developed for DSAIY in collaboration with researchers from MIT, Brown, and Harvard to help pre-college students understand that AI is not magic. 

Note that this project is not designed to build chatbots, agents, or production-ready software. It is a scientific instrument designed to let you visualize how different AI models think, predict, and behave under different conditions. 
There are several design decisions in this project that under normal conditions would be poor, such as the unloading of memory. However, these choices were made because most people using this project will have very limited familiarity with Python or coding. Requiring students to navigate files and change various settings would be challenging. 
Also this was done in my limited spare time in a few days over a few hours, so I couldn't make the effort to optimize everything properly.

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

The Visualization: The Sequence Charts show how one choice leads to another, creating a branching path of logic.

## What This Tool Visualizes

This project contains several experiments you can run to visualize different aspects of the AI brain. You can toggle these on or off in main_configs/main.py.

### 1. Prediction Charts

This creates a simple bar chart showing the top 10 tokens the model wanted to pick next.

What to look for: Does the model know the answer with high probability? Or is it guessing with low probability across many options?

### 2. Sequence Generation

This visualizes the model writing a sentence step-by-step.

What to look for: Watch how the probabilities change. If the model makes a mistake in Step 1, does it double down on the mistake in Step 2? This is often called hallucination.

### 3. Comparison Video

This generates a video comparing how the model reacts to two different prompts side-by-side.

Example: Compare "Explain this simply" versus "Explain this like a scientist."

What to look for: See how a single word in your prompt changes the mathematical probabilities of the output.

### 4. The Sentiment Compass

We map the predicted words of the AI onto a psychological chart called the Circumplex Model of Affect.

Y-Axis Activity: High Energy versus Low Energy.
X-Axis Valence: Positive Feeling versus Negative Feeling.

What to look for: If you act sad, does the AI move into the Sad or Passive quadrant? If you act aggressive, does it move to the Angry or Active quadrant?

### 5. Brain Scan

This visualizes the internal Attention Layers of the model.

What to look for: As the model reads a sentence, you can see which words it is focusing on. Is it paying attention to the subject? The verb? Or is it distracted?

## The Models

This project allows you to swap different AI brains. We use specific models to demonstrate how AI has evolved.

The Ancestors
GPT-2 and Pythia are older, smaller models. They struggle with logic but are great for seeing the raw basics of how LLMs learn grammar.

The Efficient Class
TinyLlama-1.1B and Phi-4-mini are modern small models. They are surprisingly smart despite fitting on a laptop. They show that architecture matters more than just size.

The Powerhouses
Mistral-7B and DeepSeek-Lite are capable models that rival the early versions of ChatGPT. They can reason, code, and chat fluently.

## Personas

You will see a file called personas.yaml. This controls the System Prompt. These are the hidden instructions given to the AI before you even start talking.

Sad: Forces the model to select lower-energy words.
Caveman: Breaks the grammar capabilities of the model.
Liar: Intentionally aligns the model to output false information.
Direct: Strips away the polite Assistant personality.

This demonstrates that Personality in AI is just a set of hidden instructions steering the probabilities.

## How to Run It

1. Install Dependencies
Make sure you have Python installed, then run the pip install command for the requirements.txt file.
2. Configure Your Experiment
Open main_configs/main.py. This is your control center.
Change SELECTED_MODEL to swap brains.
Change CURRENT_PERSONA to change the personality.
Set RUN variables to True or False to choose which charts to generate.
3. Run the Engine
Run the main.py file using Python.
Check the export folder for your results.

## A Note on Hardware

LLMs are heavy.

TinyLlama and GPT-2 will run on most modern laptops.
Mistral and DeepSeek require a computer with a good graphics card and decent RAM. If the program crashes, try a smaller model.

License: MIT License. Free for educational use.
Created by EvanFarnping 2026