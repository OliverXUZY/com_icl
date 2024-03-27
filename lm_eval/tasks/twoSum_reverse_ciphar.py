import regex
import string

import datasets
from lm_eval.base import rf, Task
from lm_eval.metrics import mean, matthews_corrcoef, f1_score, yesno
from lm_eval.utils import general_detokenize

import evaluate
exact_match = evaluate.load("exact_match")

class equation(Task):
    VERSION = 0
    DATASET_PATH = "json"
    DATASET_NAME = None

    cache_dir = "./data/equation/cache"
    train_files = None
    test_files = None

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
       
        testset = datasets.load_dataset(
            path=self.DATASET_PATH,
            name=self.DATASET_NAME,
            data_files=self.test_files,
            cache_dir=cache_dir,
            download_mode=download_mode,
            split = "train" # zhuoyan: this dataset does not have train/val/test split, only the dataset object
        )

        trainset = datasets.load_dataset(
            path=self.DATASET_PATH,
            name=self.DATASET_NAME,
            data_files=self.train_files,
            cache_dir=cache_dir,
            download_mode=download_mode,
            split = "train" # zhuoyan: this dataset does not have train/val/test split, only the dataset object
        )

        self.dataset = datasets.DatasetDict({
            "train": trainset,
            "validation": testset
        })

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(self.dataset["train"])
        return self._training_docs

    def validation_docs(self):
        return self.dataset["validation"]

    def doc_to_text(self, doc):
        return f"input: {doc['input']}\noutput:"


    def doc_to_target(self, doc):
        return doc["output"]

    def construct_requests(self, doc, ctx):
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.
        :param doc:
                The document as returned from training_docs, validation_docs, or test_docs.
        :param ctx: str
                The context string, generated by fewshot_context. This includes the natural
                language description, as well as the few shot examples, and the question
                part of the document for `doc`.
        """
        continuation = rf.greedy_until(ctx, {"until": ["\n"]})
        return continuation

    def _normalize_answer(self, text):
        # strip whitespace
        if len(text) > 0 and text[0] == " ":
            # print(f"text =={text}==")
            text = text.strip()

        return text

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        continuation = self._normalize_answer(results[0])
        answers = doc["output"]

        # print(f"continuation:  =={continuation}==")
        # print(f"answers: =={answers}==")

        preds = continuation.split(" ")
        refs = answers.split(" ")

        # Ensure both lists are of the same length by appending empty strings or take subset
        if len(refs) > len(preds):
            preds.extend([""] * (len(refs) - len(preds)))
        elif len(preds) > len(refs):
            preds = preds[:len(refs)]  # Slicing preds to match the length of refs
        
        results = exact_match.compute(references=refs, predictions=preds)
        
        return {"acc": results['exact_match']}

    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metrics
        """
        return {
            "acc": mean,
        }

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are
            whether a higher value of the submetric is better
        """
        return {
            "acc": True,
        }


class reverse(equation):
    train_files = 'data/twoSum_reverse_cipher/reverse_train.json'
    test_files = 'data/twoSum_reverse_cipher/reverse_test.json'


class twoSum(equation):
    train_files = 'data/twoSum_reverse_cipher/twoSum_train.json'
    test_files = 'data/twoSum_reverse_cipher/twoSum_test.json'

class reverse_twoSum(equation):
    train_files = 'data/twoSum_reverse_cipher/compose_train.json'
    test_files = 'data/twoSum_reverse_cipher/compose_test.json'
    task_1_files = 'data/twoSum_reverse_cipher/reverse_train.json'
    task_2_files = 'data/twoSum_reverse_cipher/two_sum_train.json'

    def __init__(self, data_dir=None, cache_dir=None, download_mode=None):
        self.download(data_dir, cache_dir, download_mode)
        self._training_docs = None
        self._fewshot_docs = None
        self._task1_training_docs = None
        self._task2_training_docs = None

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        testset = datasets.load_dataset(
            path=self.DATASET_PATH,
            name=self.DATASET_NAME,
            data_files=self.test_files,
            cache_dir=cache_dir,
            download_mode=download_mode,
            split = "train" # zhuoyan: this dataset does not have train/val/test split, only the dataset object
        )

        trainset = datasets.load_dataset(
            path=self.DATASET_PATH,
            name=self.DATASET_NAME,
            data_files=self.train_files,
            cache_dir=cache_dir,
            download_mode=download_mode,
            split = "train" # zhuoyan: this dataset does not have train/val/test split, only the dataset object
        )

        task_1_set = datasets.load_dataset(
            path=self.DATASET_PATH,
            name=self.DATASET_NAME,
            data_files=self.task_1_files,
            cache_dir=cache_dir,
            download_mode=download_mode,
            split = "train" # zhuoyan: this dataset does not have train/val/test split, only the dataset object
        )

        task_2_set = datasets.load_dataset(
            path=self.DATASET_PATH,
            name=self.DATASET_NAME,
            data_files=self.task_2_files,
            cache_dir=cache_dir,
            download_mode=download_mode,
            split = "train" # zhuoyan: this dataset does not have train/val/test split, only the dataset object
        )

        self.dataset = datasets.DatasetDict({
            "train": trainset,
            "validation": testset,
            "task1": task_1_set,
            "task2": task_2_set
        })

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(self.dataset["train"])
        return self._training_docs
    

    def validation_docs(self):
        return self.dataset["validation"]
    
    def fewshot_examples(self, k, rnd):
        if self._task1_training_docs is None:
                self._task1_training_docs = list(self.dataset["task1"])
        
        if self._task2_training_docs is None:
                self._task2_training_docs = list(self.dataset["task2"])
        
        retval = rnd.sample(self._task1_training_docs, k) + rnd.sample(self._task2_training_docs, k)
        rnd.shuffle(retval)
        return retval

class reverse_twoSum_compose_incontext(reverse_twoSum):
    def fewshot_examples(self, k, rnd):
        if self._training_docs is None:
            self._training_docs = list(self.training_docs())

        compose = rnd.sample(self._training_docs, k)

        if self._task1_training_docs is None:
                self._task1_training_docs = list(self.dataset["task1"])
        
        if self._task2_training_docs is None:
                self._task2_training_docs = list(self.dataset["task2"])
        
        retval = rnd.sample(self._task1_training_docs, k) + rnd.sample(self._task2_training_docs, k) + compose
        rnd.shuffle(retval)

        return retval
        
# ================================================

class ciphar(equation):
    train_files = 'data/twoSum_reverse_cipher/ciphar_train.json'
    test_files = 'data/twoSum_reverse_cipher/ciphar_test.json'


class ciphar_twoSum(reverse_twoSum):
    train_files = 'data/twoSum_reverse_cipher/twoSum_ciphar_train.json'
    test_files = 'data/twoSum_reverse_cipher/twoSum_ciphar_test.json'
    task_1_files = 'data/twoSum_reverse_cipher/ciphar_train.json'
    task_2_files = 'data/twoSum_reverse_cipher/two_sum_train.json'

    def fewshot_examples(self, k, rnd):
        if self._task1_training_docs is None:
                self._task1_training_docs = list(self.dataset["task1"])
        
        if self._task2_training_docs is None:
                self._task2_training_docs = list(self.dataset["task2"])
        
        retval = rnd.sample(self._task1_training_docs, k) + rnd.sample(self._task2_training_docs, k)
        rnd.shuffle(retval)
        return retval
    
    def process_results(self, doc, results):
        continuation = self._normalize_answer(results[0])
        answers = doc["output"]

        # print(f"continuation:  =={continuation}==")
        # print(f"answers: =={answers}==")

        preds = continuation.split(" ")
        refs = answers.split(" ")
        # print("preds: ", preds)
        # print("refs: ", refs)

        # assert False

        
        return {"acc":  float(preds[0] in refs)}


class ciphar_twoSum_compose_incontext(ciphar_twoSum):
    def fewshot_examples(self, k, rnd):
        if self._training_docs is None:
            self._training_docs = list(self.training_docs())

        compose = rnd.sample(self._training_docs, k)

        if self._task1_training_docs is None:
                self._task1_training_docs = list(self.dataset["task1"])
        
        if self._task2_training_docs is None:
                self._task2_training_docs = list(self.dataset["task2"])
        
        retval = rnd.sample(self._task1_training_docs, k) + rnd.sample(self._task2_training_docs, k) + compose
        rnd.shuffle(retval)

        return retval
    

