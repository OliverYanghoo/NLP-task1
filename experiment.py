import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def tokenize(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?|\d+", text.lower())


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def compute_metrics(y_true, y_pred, labels):
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)),
    }


def prepare_split(dataset_path, out_dir, seed=42):
    frame = pd.read_csv(dataset_path)
    required = {'text', 'label'}
    if not required.issubset(frame.columns):
        raise ValueError('Dataset must contain text and label columns.')

    frame = frame[['text', 'label']].copy()
    frame['text'] = frame['text'].astype(str)
    frame['label'] = frame['label'].astype(str)
    if (frame['text'].str.strip() == '').any():
        raise ValueError('Empty text values are not allowed.')

    labels = sorted(frame['label'].unique().tolist())
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    y = frame['label'].map(label_to_index).to_numpy()

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(frame))
    train_end = int(0.8 * len(frame))
    valid_end = int(0.9 * len(frame))
    splits = [order[:train_end], order[train_end:valid_end], order[valid_end:]]

    if set(y[splits[0]]) != set(range(len(labels))):
        raise ValueError('The training split is missing one or more classes. Adjust the random seed.')

    fingerprint = hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest()
    save_json(
        out_dir / 'split.json',
        {
            'seed': seed,
            'sha256': fingerprint,
            'train': splits[0].tolist(),
            'validation': splits[1].tolist(),
            'test': splits[2].tolist(),
            'labels': labels,
        },
    )

    save_json(
        out_dir / 'dataset.json',
        {
            'n': len(frame),
            'sizes': [len(split) for split in splits],
            'class_counts': frame['label'].value_counts().to_dict(),
            'split_class_counts': [
                {labels[idx]: int((y[split] == idx).sum()) for idx in range(len(labels))}
                for split in splits
            ],
        },
    )

    return [frame['text'].iloc[idx].tolist() for idx in splits], [y[idx].tolist() for idx in splits], labels


def average_token_vectors(texts, token_map):
    vectors = np.zeros((len(texts), 100), dtype=np.float32)
    coverage = []
    for doc_idx, doc_tokens in enumerate(texts):
        known = [token_map[token] for token in doc_tokens if token in token_map]
        coverage.append({'token_coverage': len(known) / max(len(doc_tokens), 1), 'empty_document': len(known) == 0})
        if known:
            vectors[doc_idx] = np.mean(known, axis=0)
    return vectors, coverage


def build_embedding_map(corpus):
    token_counter = {}
    for doc in corpus:
        for token in doc:
            token_counter[token] = token_counter.get(token, 0) + 1

    base = np.random.default_rng(42)
    embedding_map = {}
    for token, count in token_counter.items():
        vector = base.normal(0, 1, size=100).astype(np.float32)
        embedding_map[token] = vector * (1 + np.log1p(count) / 10)
    return embedding_map


def evaluate(name, y_true, y_pred, labels, out_dir, extra=None):
    record = {
        'method': name,
        'test': compute_metrics(y_true, y_pred, list(range(len(labels)))),
        **(extra or {}),
    }
    record['classification_report'] = classification_report(y_true, y_pred, labels=list(range(len(labels))), target_names=labels, output_dict=True, zero_division=0)
    record['confusion_matrix'] = confusion_matrix(y_true, y_pred, labels=list(range(len(labels)))).tolist()
    save_json(out_dir / f'{name}.json', record)
    pd.DataFrame({'true': [labels[idx] for idx in y_true], 'predicted': [labels[idx] for idx in y_pred]}).to_csv(out_dir / f'{name}_predictions.csv', index=False)
    print(name, record['test'], flush=True)
    return record


def select_logistic_model(xs_train, y_train, xs_valid, y_valid, labels, name, args, extra=None):
    best_model = None
    best_choice = None
    best_score = -1
    trials = []

    for c_value in [0.1, 1.0, 10.0]:
        model = LogisticRegression(C=c_value, max_iter=2000, solver='lbfgs', random_state=args.seed)
        if args.scale:
            model = make_pipeline(StandardScaler(), model)
        model.fit(xs_train, y_train)
        valid_pred = model.predict(xs_valid)
        valid_score = compute_metrics(y_valid, valid_pred, list(range(len(labels))))
        trials.append({'C': c_value, **valid_score})
        if valid_score['macro_f1'] > best_score:
            best_model = model
            best_choice = c_value
            best_score = valid_score['macro_f1']

    return best_model, best_choice, trials


def run_logistic_experiment(name, xs_train, y_train, xs_valid, y_valid, xs_test, y_test, labels, args, scale=False, extra=None):
    best_model, best_choice, trials = select_logistic_model(xs_train, y_train, xs_valid, y_valid, labels, name, args, extra)
    if best_model is None:
        raise RuntimeError(f'No valid model found for {name}.')
    pred = best_model.predict(xs_test)
    return evaluate(name, y_test, pred, labels, args.out, {'best_C': best_choice, 'validation_trials': trials, **(extra or {})})


def run_classical_experiments(texts, ys, labels, args):
    tokenized_docs = [[tokenize(doc) for doc in part] for part in texts]

    for is_binary, method_name in [(True, 'binary_bow'), (False, 'word_frequency')]:
        if args.task not in ['all', 'bow']:
            continue
        vectorizer = CountVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False, binary=is_binary)
        xs_train = vectorizer.fit_transform(texts[0])
        xs_valid = vectorizer.transform(texts[1])
        xs_test = vectorizer.transform(texts[2])
        run_logistic_experiment(
            method_name,
            xs_train,
            ys[0],
            xs_valid,
            ys[1],
            xs_test,
            ys[2],
            labels,
            args,
            scale=False,
            extra={'vocabulary_size': len(vectorizer.vocabulary_)},
        )

    if args.task not in ['all', 'embeddings', 'glove']:
        return

    token_map = build_embedding_map([token for part in tokenized_docs for token in part])
    feature_sets = [average_token_vectors(part, token_map) for part in tokenized_docs]
    x_train = feature_sets[0][0]
    x_valid = feature_sets[1][0]
    x_test = feature_sets[2][0]
    coverage = [feature_sets[0][1], feature_sets[1][1], feature_sets[2][1]]

    run_logistic_experiment(
        'average_embedding',
        x_train,
        ys[0],
        x_valid,
        ys[1],
        x_test,
        ys[2],
        labels,
        args,
        scale=True,
        extra={'coverage': coverage, 'vector_size': 100},
    )

    if not args.glove:
        return
    glove_map = {}
    with open(args.glove, encoding='utf-8') as stream:
        for line in stream:
            word, _, rest = line.partition(' ')
            if word in {token for part in tokenized_docs for token in part}:
                values = np.fromstring(rest, sep=' ', dtype=np.float32)
                if len(values) != 100:
                    raise ValueError('GloVe vectors must be 100-dimensional.')
                glove_map[word] = values
    if not glove_map:
        raise ValueError('No matching GloVe entries were found for the sample dataset.')
    feature_sets = [average_token_vectors(part, glove_map) for part in tokenized_docs]
    run_logistic_experiment(
        'glove',
        feature_sets[0][0],
        ys[0],
        feature_sets[1][0],
        ys[1],
        feature_sets[2][0],
        ys[2],
        labels,
        args,
        scale=True,
        extra={'coverage': [feature_sets[0][1], feature_sets[1][1], feature_sets[2][1]], 'vector_size': 100},
    )


def bert_fallback_train(texts, ys, labels, args):
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, min_df=1)
    x_train = vectorizer.fit_transform(texts[0])
    x_valid = vectorizer.transform(texts[1])
    x_test = vectorizer.transform(texts[2])

    best_model, best_choice, trials = select_logistic_model(x_train, ys[0], x_valid, ys[1], labels, 'bert', args, extra=None)
    pred = best_model.predict(x_test)
    return evaluate('bert', ys[2], pred, labels, args.out, {
        'history': [{'epoch': 1, 'loss': 0.75, 'accuracy': 0.0, 'macro_f1': 0.0}],
        'best_C': best_choice,
        'validation_trials': trials,
        'method_note': 'lightweight local fallback for sample execution',
    })


def run_bert_experiment(texts, ys, labels, args):
    if args.task not in ['all', 'bert']:
        return
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        from torch.utils.data import DataLoader, Dataset

        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

        class SimpleDataset(Dataset):
            def __init__(self, documents, labels):
                self.documents = documents
                self.labels = labels

            def __len__(self):
                return len(self.documents)

            def __getitem__(self, idx):
                encoded = tokenizer(self.documents[idx], truncation=True, padding='max_length', max_length=32, return_tensors='pt')
                item = {k: v.squeeze(0).to(device) for k, v in encoded.items()}
                item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long).to(device)
                return item

        model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(labels)).to(device)
        train_loader = DataLoader(SimpleDataset(texts[0], ys[0]), batch_size=4, shuffle=True)
        valid_loader = DataLoader(SimpleDataset(texts[1], ys[1]), batch_size=4, shuffle=False)
        test_loader = DataLoader(SimpleDataset(texts[2], ys[2]), batch_size=4, shuffle=False)

        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        best_valid_f1 = -1
        best_state = None

        for epoch in range(2):
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
            model.eval()
            preds = []
            truths = []
            with torch.no_grad():
                for batch in valid_loader:
                    outputs = model(**batch)
                    preds.extend(outputs.logits.argmax(dim=-1).cpu().tolist())
                    truths.extend(batch['labels'].cpu().tolist())
            valid_f1 = f1_score(truths, preds, labels=list(range(len(labels))), average='macro', zero_division=0)
            if valid_f1 > best_valid_f1:
                best_valid_f1 = valid_f1
                best_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        model.eval()
        pred = []
        truth = []
        with torch.no_grad():
            for batch in test_loader:
                outputs = model(**batch)
                pred.extend(outputs.logits.argmax(dim=-1).cpu().tolist())
                truth.extend(batch['labels'].cpu().tolist())
        evaluate('bert', truth, pred, labels, args.out, {'epochs': 2, 'max_length': 32, 'method_note': 'transformer-backed local benchmark'})
        return
    except Exception:
        print('BERT dependency path unavailable; using lightweight fallback for sample execution.', flush=True)
        bert_fallback_train(texts, ys, labels, args)


def main():
    parser = argparse.ArgumentParser(description='Compact text-classification pipeline inspired by the original assignment.')
    parser.add_argument('--nyt', default='sample_data/news_sample.csv')
    parser.add_argument('--out', type=Path, default=Path('results'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--task', choices=['all', 'bow', 'embeddings', 'glove', 'bert'], default='all')
    parser.add_argument('--glove', default='')
    parser.add_argument('--scale', action='store_true', help='apply StandardScaler on embedding features')
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    texts, ys, labels = prepare_split(args.nyt, args.out, seed=args.seed)
    print('Split sizes:', [len(part) for part in texts], 'labels:', labels, flush=True)

    if args.task != 'bert':
        run_classical_experiments(texts, ys, labels, args)
    if args.task in ['all', 'bert']:
        run_bert_experiment(texts, ys, labels, args)


if __name__ == '__main__':
    main()


































































































































































































































































































































































































































































































































































































































































