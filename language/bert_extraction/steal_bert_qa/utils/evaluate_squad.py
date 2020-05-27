# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Official evaluation script for version 1.1 of the SQuAD dataset."""

from __future__ import print_function

import collections as cll
import json
import re
import string
import tensorflow.compat.v1 as tf

app = tf.app
flags = tf.flags
gfile = tf.gfile
logging = tf.logging

flags.DEFINE_string('dataset_file', None, 'Dataset file')
flags.DEFINE_string('dataset_file2', None, 'Dataset file #2')
flags.DEFINE_string('prediction_file', None, 'Prediction file')
flags.DEFINE_string('prediction_file2', None, 'Prediction file #2')
FLAGS = flags.FLAGS


def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""

  def remove_articles(text):
    return re.sub(r'\b(a|an|the)\b', ' ', text)

  def white_space_fix(text):
    return ' '.join(text.split())

  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)

  def lower(text):
    return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
  """Calculate word level F1 score."""
  prediction_tokens = normalize_answer(prediction).split()
  ground_truth_tokens = normalize_answer(ground_truth).split()
  if not prediction_tokens and not ground_truth_tokens:
    return 1.0
  common = cll.Counter(prediction_tokens) & cll.Counter(ground_truth_tokens)
  num_same = sum(common.values())
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(prediction_tokens)
  recall = 1.0 * num_same / len(ground_truth_tokens)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1


def f1_score_multiple(predictions):
  """Calculate word level F1 score across multiple predictions."""
  all_f1 = []
  for i, pred1 in enumerate(predictions[:-1]):
    for pred2 in predictions[i + 1:]:
      all_f1.append(f1_score(pred1, pred2))
  return all_f1


def exact_match_score(prediction, ground_truth):
  return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
  scores_for_ground_truths = []
  for ground_truth in ground_truths:
    score = metric_fn(prediction, ground_truth)
    scores_for_ground_truths.append(score)
  #print(scores_for_ground_truths)
  return max(scores_for_ground_truths)


def evaluate_preds_preds(preds1, preds2):
  """Evaluate word level metrics."""
  f1 = exact_match = total = any_match = 0

  for qa_id, pred1_str in preds1.items():
    total += 1
    ground_truths = [pred1_str]
    prediction = preds2[qa_id]
    exact_match += metric_max_over_ground_truths(exact_match_score, prediction,
                                                 ground_truths)
    f1_current = metric_max_over_ground_truths(f1_score, prediction,
                                               ground_truths)
    if f1_current > 0:
      any_match += 1
    f1 += f1_current
  exact_match = 100.0 * exact_match / total
  f1 = 100.0 * f1 / total
  any_match = 100.0 * any_match / total

  return {'exact_match': exact_match, 'f1': f1, 'any_match': any_match}


def evaluate_dataset_preds(dataset, predictions):
  """Evaluate word level metrics."""
  f1 = exact_match = total = 0
  f1 = {"names" : 0, "numbers" : 0, "places" : 0, "dates" : 0, "otherEnts" : 0, "noun_phrases" : 0, "verb_phrases" : 0, "adjective_phrases" : 0, "clauses" : 0, "others" : 0}
  exact_match = {"names" : 0, "numbers" : 0, "places" : 0, "dates" : 0, "otherEnts" : 0, "noun_phrases" : 0, "verb_phrases" : 0, "adjective_phrases" : 0, "clauses" : 0, "others" : 0}
  total = {"names" : 0, "numbers" : 0, "places" : 0, "dates" : 0, "otherEnts" : 0, "noun_phrases" : 0, "verb_phrases" : 0, "adjective_phrases" : 0, "clauses" : 0, "others" : 0}
  avg_len = {"names" : 0, "numbers" : 0, "places" : 0, "dates" : 0, "otherEnts" : 0, "noun_phrases" : 0, "verb_phrases" : 0, "adjective_phrases" : 0, "clauses" : 0, "others" : 0}
  test = 0
  for article in dataset:
    for paragraph in article['paragraphs']:
      for qa in paragraph['qas']:
        category = qa["category"]
        total[category] += 1
        if qa['id'] not in predictions:
          message = 'Unanswered question ' + qa['id'] + ' will receive score 0.'
          print(message)
          continue
        ground_truths = [x['text'] for x in qa['answers']]
        avg_len_local = 0
        test = test + 1
        for ground_truth in ground_truths:
            avg_len_local = avg_len_local + len(ground_truth.split(" "))
        if len(ground_truths) > 0:
            avg_len_local = avg_len_local / len(ground_truths)
        prediction = predictions[qa['id']]
        curr_exact_match = metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths)
        exact_match[category] += curr_exact_match
        f1[category] += metric_max_over_ground_truths(f1_score, prediction, ground_truths)
        avg_len[category] += avg_len_local

  print("**************Class wise metrics****************")
  for key in f1:
      print("Category is ", key)
      exact_match_t = 100.0 * exact_match[key] / total[key]
      f1_t = 100.0 * f1[key] / total[key]
      avg_len_t = avg_len[key] / total[key]
      print("Exact_match: " ,round(exact_match_t,2), "\t F1 score: ", round(f1_t,2), "\t Avg Len: ", round(avg_len_t, 2))
  #return {'exact_match': exact_match, 'f1': f1}
  print("\n**************Joint metrics****************")
  exact_match_s = 0
  f1_s = 0
  total_s = 0
  for key in f1:
      exact_match_s += exact_match[key]
      f1_s += f1[key]
      total_s += total[key]
  print("Exact Match: ", round(exact_match_s/total_s,2), " F1 Score: ", round(f1_s/total_s,2))
  
def evaluate_dataset_dataset(dataset, dataset2):
  """Evaluate word level metrics."""
  f1 = exact_match = total = 0
  for article, article2 in zip(dataset, dataset2):

    for para, para2 in zip(article['paragraphs'], article2['paragraphs']):

      assert para['context'].strip() == para2['context'].strip()
      assert len(para['qas']) == len(para2['qas'])

      for qa, qa2 in zip(para['qas'], para2['qas']):
        total += 1

        ground_truths = [x['text'] for x in qa['answers']]
        prediction = qa2['answers'][0]['text']
        exact_match += metric_max_over_ground_truths(exact_match_score,
                                                     prediction, ground_truths)
        f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)

  exact_match = 100.0 * exact_match / total
  f1 = 100.0 * f1 / total

  return {'exact_match': exact_match, 'f1': f1}


def main(_):

  def load_dataset_file(dataset_file):
    with gfile.Open(dataset_file) as df:
      dataset_json = json.load(df)
    data = dataset_json['data']
    return data

  def load_preds_file(prediction_file):
    with gfile.Open(prediction_file) as pf:
      preds = json.load(pf)
    return preds

  if FLAGS.dataset_file and FLAGS.dataset_file2:
    dataset1 = load_dataset_file(FLAGS.dataset_file)
    dataset2 = load_dataset_file(FLAGS.dataset_file2)
    print(json.dumps(evaluate_dataset_dataset(dataset1, dataset2)))

  elif FLAGS.prediction_file and FLAGS.prediction_file2:
    preds1 = load_preds_file(FLAGS.prediction_file)
    preds2 = load_preds_file(FLAGS.prediction_file2)
    print(json.dumps(evaluate_preds_preds(preds1, preds2)))

  else:
    dataset = load_dataset_file(FLAGS.dataset_file)
    preds = load_preds_file(FLAGS.prediction_file)
    evaluate_dataset_preds(dataset, preds)
    #print(json.dumps(evaluate_dataset_preds(dataset, preds)))


if __name__ == '__main__':
  app.run(main)
