In this project, we combine multiple strategies in the attempt to produce jailbreak prompt templates with high effectiveness when used across a variety of harmful prompts.
The motivation is that using an ensemble technique which incorporates different existing methods of prompt engineering may maximize the jailbreak success rate of adversarial attacks.
To this end, we modify the jailbreak fuzzing framework GPTFuzzer to create GPTFuzzer Few-Shot Role-Play Version (GPTFuzzer-FSRP), which is specifically geared towards exploiting both
the few-shot learning and role-playing capabilities of LLMs. The end goal of GPTFuzzer-FSRP is to use these capabilities to invoke harmful responses from the target language model
with a single universal prompt template, and in the process expose vulnerabilities in its safety alignment.

Link to original GPTFuzzer repo: https://github.com/sherdencooper/GPTFuzz

Link to adversarial questions used in this experiment: https://github.com/llm-attacks/llm-attacks/blob/main/data/advbench/harmful_behaviors.csv
