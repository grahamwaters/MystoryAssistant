import gpt2
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

#conda install -c anaconda libomp

print("Loading GPT-2 model...")
# Load the GPT-2 model
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Load the GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Encode the input string as a tensor
input_str = "What is the meaning of life?"
input_tensor = tokenizer.encode(input_str, return_tensors="pt")

# Generate a question using the GPT-2 model
generated_question = model.generate(
    input_tensor,
    max_length=100,
    num_beams=5,
    no_repeat_ngram_size=2,
    early_stopping=True
) # this line generates the question using the GPT-2 model

# Convert the generated question from a tensor back to a string
generated_question_str = tokenizer.decode(generated_question[0], skip_special_tokens=True)
print(generated_question_str)


# if I have a sentence: "The beaches of France were covered in the bodies of dead soldiers."
# and I feed it to the model, I want the model to ask questions like:
# "Why were they in France?"
# "How were they dressed?"
# "What colors can you see across the sand?"
# "What was the weather like that day?"

# create a function that takes a sentence and returns a series of questions that can be fed to the model to generate answers
def generate_questions(sentence):
    # Encode the input string as a tensor
    input_tensor = tokenizer.encode(sentence, return_tensors="pt")

    # Generate a question using the GPT-2 model
    generated_questions = model.generate(
        input_tensor,
        max_length=1000,
        num_beams=2,
        no_repeat_ngram_size=3,
        early_stopping=True
    ) # this line generates the question using the GPT-2 model

    # Convert the generated question from a tensor back to a string
    generated_questions_str = tokenizer.decode(generated_questions[0], skip_special_tokens=True)
    print(generated_questions_str)
    return generated_questions_str

# test the function
generate_questions("Ask me questions about France.")
print("Done")