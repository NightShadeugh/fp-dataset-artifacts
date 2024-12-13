import json
import random
import ssl
import nltk
import os

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def download_nltk_data():
    nltk_download_dir = os.path.join(os.path.expanduser('~'), 'nltk_data')
    os.makedirs(nltk_download_dir, exist_ok=True)
    nltk.download('wordnet', download_dir=nltk_download_dir)
    nltk.download('punkt', download_dir=nltk_download_dir)
    nltk.data.path.append(nltk_download_dir)
download_nltk_data()

from nltk.corpus import wordnet
from transformers import MarianMTModel, MarianTokenizer

class DataAugmenter:
    def __init__(self):
        self.tokenizer, self.model, self.reverse_tokenizer, self.reverse_model = self._load_models()

    def _load_models(self):
        model_name = 'Helsinki-NLP/opus-mt-en-fr'
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        
        reverse_model_name = 'Helsinki-NLP/opus-mt-fr-en'
        reverse_tokenizer = MarianTokenizer.from_pretrained(reverse_model_name)
        reverse_model = MarianMTModel.from_pretrained(reverse_model_name)
        
        return tokenizer, model, reverse_tokenizer, reverse_model

    def _synonym_replacement(self, text, n=1):
        """
        Replace words with their synonyms
        """
        words = text.split()
        new_words = words.copy()
        
        for _ in range(n):
            if len(words) > 0:
                random_word_index = random.randint(0, len(words) - 1)
                synonyms = self._get_synonyms(words[random_word_index])
                
                if synonyms:
                    new_words[random_word_index] = random.choice(synonyms)
        
        return ' '.join(new_words)

    def _get_synonyms(self, word):
        """
        Get synonyms for a word using WordNet
        """
        synonyms = set()
        try:
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    if lemma.name().lower() != word.lower():
                        synonyms.add(lemma.name().replace('_', ' '))
        except Exception as e:
            print(f"Could not find synonyms for word '{word}': {e}")
        return list(synonyms)

    def _back_translation(self, text):
        """
        Perform back translation
        """
        try:
            translated = self.model.generate(**self.tokenizer(text, return_tensors="pt", padding=True, truncation=True))
            fr_text = self.tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
            translated_back = self.reverse_model.generate(**self.reverse_tokenizer(fr_text, return_tensors="pt", padding=True, truncation=True))
            back_translated_text = self.reverse_tokenizer.batch_decode(translated_back, skip_special_tokens=True)[0]
            
            return back_translated_text
        except Exception as e:
            print(f"Back translation failed: {e}")
            return text

    def _semantic_similarity_augmentation(self, entry):
        """
        Create semantically similar variations
        """
        augmented_entries = []
        premise = entry['premise']
        hypothesis = entry['hypothesis']
        label = entry['label']

        augmented_premise = self._synonym_replacement(premise)
        augmented_hypothesis = self._synonym_replacement(hypothesis)
        augmented_entries.append({
            'premise': augmented_premise,
            'hypothesis': hypothesis,
            'label': label
        })
        augmented_entries.append({
            'premise': premise,
            'hypothesis': augmented_hypothesis,
            'label': label
        })

        return augmented_entries

    def augment_data(self, dataset):
        """
        Comprehensive data augmentation
        """
        augmented_data = []

        for entry in dataset:
            augmented_data.append(entry)

            argumentative_variants = self._generate_argumentative_variants(entry)
            augmented_data.extend(argumentative_variants)
            semantic_variants = self._semantic_similarity_augmentation(entry)
            augmented_data.extend(semantic_variants)
            back_translated_premise = self._back_translation(entry['premise'])
            back_translated_hypothesis = self._back_translation(entry['hypothesis'])

            augmented_data.append({
                'premise': back_translated_premise,
                'hypothesis': entry['hypothesis'],
                'label': entry['label']
            })
            augmented_data.append({
                'premise': entry['premise'],
                'hypothesis': back_translated_hypothesis,
                'label': entry['label']
            })

        return augmented_data

    def _generate_argumentative_variants(self, entry):
        """
        Generate argumentative variants with more nuanced language
        """
        premise = entry['premise']
        hypothesis = entry['hypothesis']
        label = entry['label']

        augmented_entries = []

        if label == 0:  # Entailment
            augmented_entries.extend([
                {
                    'premise': premise,
                    'hypothesis': f"Given the context, {hypothesis} is undeniably true.",
                    'label': 0
                },
                {
                    'premise': premise,
                    'hypothesis': f"The evidence strongly supports the claim that {hypothesis}.",
                    'label': 0
                }
            ])
        elif label == 1:  # Neutral
            augmented_entries.extend([
                {
                    'premise': premise,
                    'hypothesis': f"While not directly implied, {hypothesis} remains a plausible scenario.",
                    'label': 1
                },
                {
                    'premise': premise,
                    'hypothesis': f"The given information neither confirms nor refutes {hypothesis}.",
                    'label': 1
                }
            ])
        elif label == 2:  # Contradiction
            augmented_entries.extend([
                {
                    'premise': premise,
                    'hypothesis': f"The premise fundamentally contradicts the assertion that {hypothesis}.",
                    'label': 2
                },
                {
                    'premise': premise,
                    'hypothesis': f"Direct evidence suggests {hypothesis} is false.",
                    'label': 2
                }
            ])

        return augmented_entries

def main():
    # Load dataset
    with open('adversarial_set_train.jsonl', 'r') as f:
        dataset = [json.loads(line) for line in f]
    augmenter = DataAugmenter()
    augmented_data = augmenter.augment_data(dataset)
    with open('augmented_train.jsonl', 'w') as f:
        for entry in augmented_data:
            f.write(json.dumps(entry) + '\n')

    print(f"Augmented dataset created with {len(augmented_data)} entries")

if __name__ == "__main__":
    main()