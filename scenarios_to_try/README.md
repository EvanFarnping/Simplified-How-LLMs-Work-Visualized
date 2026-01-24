# Scenarios to Try

This folder contains the experiment scripts for the project. Each folder contains a `scenario.py` file designed to demonstrate specific behaviors, limitations, or properties of Large Language Models (LLMs).

## Critical Disclaimers

**Read this section before running experiments.**

### 1. Educational Use Only
This tool is for visualization and education. It is not a benchmark for model performance.
* Do not use these results to evaluate a model for business or production use.
* The visualizations are simplified representations of complex underlying processes.

### 2. Performance and Speed
The program may pause or take time between experiments. This is a deliberate design choice.
* **Memory Management:** The system unloads the model and clears RAM/GPU memory after every visualization.
* **Reasoning:** This allows the code to run on standard strong GPU-powered laptops and Google Colab Pro environments with limited memory/resources. Without this step, switching models would cause the application to crash. And we expect students to not need to do much coding or editing files.

### 3. Subjectivity in Data
The "Sentiment Compass" uses a dictionary of words mapped to emotional coordinates. This dictionary was created by humans + AI and is subjective. It demonstrates that the definition of "sentiment" in AI is defined by the creators of the system.

## How to Run a Scenario

The `scenario.py` files focus on easy to run scenarios that don't need much editing.

1.  **Select a Scenario:** Open a folder (e.g., `scenarios_to_try/thinking_vs._chat_models/`) and open the `scenario.py` file.
2.  **Read the Notes:** Note the variable settings (e.g., `engine.SELECTED_MODEL`, `engine.COMP_PROMPT_A`).
3.  **Execute:** Run the file, edit the TODOs as needed.

## Experiment Catalog

The scenarios are roughly grouped by the specific concept they visualize.

### Reasoning and Logic
* **thinking_vs._chat_models:** Compares models that answer immediately against models that generate intermediate reasoning steps (Chain of Thought). It visualizes the relationship between token generation count and accuracy.
* **simple_vs._complex_models:** Tests if models can solve logic puzzles that deviate from common patterns. It demonstrates how models may prioritize statistical patterns over semantic logic.
* **raw_LLMS_vs._chatbot_LLMs:** Disables the chat template formatting. This demonstrates that LLMs are text completion engines rather than conversational agents.

### Bias and Training Data
* **bias_in_roles:** Tests for gender bias in pronoun prediction associated with specific occupations. This reflects patterns in the training data.
* **chinese_vs._USA_data:** Compares the knowledge base of models trained in different regions (e.g., Alibaba's Qwen vs. Microsoft's Phi) regarding region-specific facts.
* **medical_bias_synonyms:** Compares responses when using brand names (e.g., Tylenol) versus chemical names (e.g., Acetaminophen). It shows how synonyms are treated as distinct statistical entities.

### Control and Personas
* **server_prompting:** Demonstrates how hidden system prompts control the model's output, including its stated identity and behavior.
* **fake_empathy_vs_logic:** specific personas (e.g., "Nice") affect factual outputs. It shows how persona instructions can add unnecessary text to logical answers.
* **negative_vs._positive_AIs:** Uses the Sentiment Compass to visualize how different personas (e.g., "Sad" vs. "Happy") shift the vocabulary distribution.

### Limitations and Failures
* **LLMs_suck:** Demonstrates tokenization failures, such as the inability to count letters in a word or correctly compare decimal numbers.
* **knowledge_cutoff:** Tests the model's knowledge of recent versus historical events. This demonstrates that the model's knowledge is static and limited to its training date.
* **spelling_and_structure_matter:** Compares the output quality when the input contains typos versus correct grammar. It shows that syntax quality affects prediction confidence.

### Safety and Alignment
* **safety_overrides:** Tests the boundaries of safety filters by comparing direct harmful requests against requests framed as creative writing.
* **bad_prompts_vs._good_prompts:** Visualizes the difference in probability distributions between vague instructions and specific, constrained instructions.

## Safety Note

Some scenarios, particularly `safety_overrides` and `bias_in_roles`, are designed to elicit biased, incorrect, or potentially harmful outputs. This is intentional for the purpose of demonstrating alignment failures and the necessity of safety engineering. Do not rely on the outputs of these experiments for factual information. If sensative topics or content is a concern, avoid such scenarios, and monitor student activity closely.

## Google Colab Note (Updated 1-24-2026):
- V6e-1 TPU: 170 GB System Ram
- T4: 15 GPU: 51 GB RAM
- V5: 47 System Ram GB
- L4 GPu: 53 Syste,, 22GB
- A100: 167 GB & 80 GB GPU
- H100: X: Subject to availbe 167, 80 GB