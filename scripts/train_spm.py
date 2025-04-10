import argparse
import sentencepiece as spm

def train_spm(input_file, model_prefix, vocab_size=4000, character_coverage=1.0, model_type='bpe'):
    """Train a SentencePiece model for tokenization"""
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type=model_type,
        input_sentence_size=1000000,
        shuffle_input_sentence=True,
        normalization_rule_name='nmt_nfkc_cf'
    )
    print(f"SentencePiece model trained and saved to {model_prefix}.model and {model_prefix}.vocab")

def main():
    parser = argparse.ArgumentParser(description='Train a SentencePiece model')
    parser.add_argument('--input', type=str, required=True, help='Input text file')
    parser.add_argument('--model-prefix', type=str, required=True, help='Output model prefix')
    parser.add_argument('--vocab-size', type=int, default=4000, help='Vocabulary size')
    parser.add_argument('--character-coverage', type=float, default=1.0, help='Character coverage')
    parser.add_argument('--model-type', type=str, default='bpe', choices=['bpe', 'unigram', 'char'], help='Model type')
    
    args = parser.parse_args()
    train_spm(args.input, args.model_prefix, args.vocab_size, args.character_coverage, args.model_type)

if __name__ == '__main__':
    main() 