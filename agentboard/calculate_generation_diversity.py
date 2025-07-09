from fast_bleu import BLEU, SelfBLEU
from rouge import Rouge
import fire
import os
import json
import evaluate
from tqdm import tqdm

def calculate_generation_diversity(generation_file, ngram=4):
    """
    Calculate the generation diversity of a generation file
    """
    # read jsonl file
    
    generations = dict()
    with open(generation_file, "r") as f:
        for line in f:
            data = json.loads(line)
            generation = data["generation"]
            index = data["task_id"]
            if index not in generations:
                generations[index] = []
            generations[index].append(generation)
    
    all_bigram_scores = []
    all_trigram_scores = []
    all_rouge_scores = []
    rouge = evaluate.load('rouge')
    
    for index in tqdm(generations, desc="Calculating diversity"):
        references = generations[index][:5]
        weights = {'bigram': (1/2., 1/2.), 'trigram': (1/3., 1/3., 1/3.)}
        self_bleu = SelfBLEU(references, weights)
        scores = self_bleu.get_score()
        
        bigram = sum(scores["bigram"])/len(references)
        trigram = sum(scores["trigram"])/len(references)
        
        rouge_scores = []
        predictions = references
        references_rouge = [[references[j] for j in range(len(references)) if j!= i] for i in range(len(references))]

        all_rouge_scores.append(rouge.compute(predictions=predictions, references=references_rouge)['rougeL'])

        
        all_bigram_scores.append(bigram)    
        all_trigram_scores.append(trigram)
    diversity = {"bigram": sum(all_bigram_scores)/len(all_bigram_scores), "trigram": sum(all_trigram_scores)/len(all_trigram_scores),
                 "rouge": sum(all_rouge_scores)/len(all_rouge_scores)}    
    
    
    print("Bigram diversity: ", diversity["bigram"])
    print("Trigram diversity: ", diversity["trigram"])
    print("Rouge diversity: ", diversity["rouge"])
    
    
    
    return diversity

def main(file_path):
    file_path = os.path.join(file_path, "humaneval_output.jsonl")
    calculate_generation_diversity(file_path)
    
if __name__ == "__main__":
    fire.Fire(main)

