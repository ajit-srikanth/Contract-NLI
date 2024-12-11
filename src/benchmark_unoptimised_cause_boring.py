import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loader import ContractNLIExample
from sentence_transformers import CrossEncoder
from argparse import ArgumentParser
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ContractLoader:
    def __init__(self, data_path, batchsize = 10):
        self.data_path = data_path
        self.data = ContractNLIExample.load(json.load(open(data_path, 'r')))
        self.batchsize = batchsize
        self.test_loader = self.load_test_loader()

    def load_test_loader(self): # booo, who wrote this shit
        qs = []
        trues = []
        for e in self.data:
            qs.append((e.context_text,e.hypothesis_text))
            trues.append(e.label.value)
        
        return DataLoader(list(zip(qs, trues)), batch_size=self.batchsize, shuffle=False)

    def benchmark(self, model, output_path, device): # lol small data so why not (will change lmao)
        correct = 0
        with open(output_path, 'w') as f:
            for i, e in tqdm(enumerate(self.data)):
                score = model.predict([(e.context_text,e.hypothesis_text)])
                label_mapping = ['Contradiction', 'Entailment', 'Not-mentioned']
                # labels = [label_mapping[score_max] for score_max in score.argmax(axis=1)]
                labels = label_mapping[score.argmax(axis=1).squeeze()]
                f.write(f"{i} {labels} {e.label}\n")
                correct += ( (abs(score.argmax(axis=1) - 2)).squeeze() == e.label.value) # abs(x - 2)  cause C E N -> N E C
            print(f"accuracy = {correct}/{len(self.data)} = {correct / len(self.data) * 100:.2f}%")

def main(args):
    datapath = args.data_path
    modelpath = args.model
    outputpath = args.output_path
    if outputpath is None:
        outputpath = 'outputs/out_' + modelpath + '.txt'
    model = CrossEncoder('cross-encoder/' + modelpath, device=device)
    data = ContractLoader(datapath, batchsize=256)
    data.benchmark(model, output_path=outputpath, device=device)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", default='dataset/contract-nli/test.json', type=str, help = "data path for testing")
    parser.add_argument("--model", default='nli-deberta-v3-small', type=str, help = "huggigface model path {if baseline} else undecided lmao")
    parser.add_argument("--model_classes", default='CEN', type=str, help="index-Class correcpondance (1st letter only) : Contradiction, Entailment, Not Mentioned") 
    parser.add_argument("--output_path", default=None, type=str, help="output file path")
    args = parser.parse_args()
    main(args)