from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Load pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
model = GPT2LMHeadModel.from_pretrained('gpt2-large')

# Fine-tune the model on your dataset
# Example code for fine-tuning:
# train_dataset = load_dataset(...)
# fine_tuned_model = fine_tune_model(model, train_dataset)

# Prompting for text generation
# prompt_text = "For this task, given a query, output the correct answer from the given options. For example, Query: The access matrix approach to protection has the difficulty that, Answer: the matrix, if stored directly, is large and can be clumsy to manage. Query: An integer c is a common divisor of two integers x and y if and only if c is a divisor of x and c is a divisor of y, choose the correct answer from [A:{-6,-2, -1, 1, 2, 6}, B:{-6, -2, -1, 0, 1, 2, 6}, C:{-6, -3, -2, -1, 1, 2, 3, 6}, D:{-6, -3, -2, -1, 0, 1, 2, 3, 6}. your output should just be one of the four A,B,C,D."



# task_description = "For this task, given a query, output the correct answer from the given options."

in_context = [ "When asked to complete the given query: A completely submerged object always displaces its own, you should output: volume of fluid"
]

question = "Similarly for the query: Among these colors, the one that has the most energy per photon is, and 4 options A. red, B. yellow-green, C. blue, D. violet. Tell me the correct answer?"

prompt_text_ = in_context[0]+" "+question       #task_description+" "+

input_ids = tokenizer.encode(prompt_text_, return_tensors='pt')

# Generate text based on the prompt
output = model.generate(input_ids, num_return_sequences=1, temperature=0.7, pad_token_id=tokenizer.eos_token_id, do_sample=True)
# Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated text:", generated_text)

# Options: [A. weight of fluid, B. volume of fluid, C. density of fluid, D. All of these]