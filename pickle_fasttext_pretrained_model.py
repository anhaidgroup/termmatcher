from gensim.models import FastText as ft
import pickle


model_name = "gensim_fasttext_pretrained_bin_model.pickle"

model = ft.load_fasttext_format("wiki.en.bin") # has to be bin

with open(model_name, "wb") as output:
    pickle.dump(model, output)

print("done")