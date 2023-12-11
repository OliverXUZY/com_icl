import regex
import string

import datasets
from lm_eval.base import rf, Task
from lm_eval.metrics import mean, matthews_corrcoef, f1_score, yesno
from lm_eval.utils import general_detokenize

import evaluate
exact_match = evaluate.load("exact_match")

class cofe(Task):
    VERSION = 0
    DATASET_PATH = "json"
    DATASET_NAME = None

    cache_dir = "./data/cofe/cache"
    data_files = 'data/cofe/raw/only_primitive_coverage.json'
    # data_dir = 'cofe/only_primitive_coverage'

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        """Downloads and returns the task dataset.
        Override this method to download the dataset from a custom API.

        :param data_dir: str
            Stores the path to a local folder containing the `Task`'s data files.
            Use this to specify the path to manually downloaded data (usually when
            the dataset is not publicly accessible).
        :param cache_dir: str
            The directory to read/write the `Task` dataset. This follows the
            HuggingFace `datasets` API with the default cache directory located at:
                `~/.cache/huggingface/datasets`
            NOTE: You can change the cache location globally for a given process
            by setting the shell environment variable, `HF_DATASETS_CACHE`,
            to another directory:
                `export HF_DATASETS_CACHE="/path/to/another/directory"`
        :param download_mode: datasets.DownloadMode
            How to treat pre-existing `Task` downloads and data.
            - `datasets.DownloadMode.REUSE_DATASET_IF_EXISTS`
                Reuse download and reuse dataset.
            - `datasets.DownloadMode.REUSE_CACHE_IF_EXISTS`
                Reuse download with fresh dataset.
            - `datasets.DownloadMode.FORCE_REDOWNLOAD`
                Fresh download and fresh dataset.
        """
        self.dataset = datasets.load_dataset(
            path=self.DATASET_PATH,
            name=self.DATASET_NAME,
            data_files=self.data_files,
            cache_dir=self.cache_dir,
            download_mode=download_mode,
            split = "train" # zhuoyan: this dataset does not have train/val/test split, only the dataset object
        )

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def validation_docs(self):
        return self.dataset

    def doc_to_text(self, doc):
        return f"{doc['context']}\noutput:"


    def doc_to_target(self, doc):
        return doc["ground_truth"]

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
        answers = doc["ground_truth"]

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

class Deeper_Nesting(cofe):
    data_files = 'data/cofe/only_primitive_coverage/Deeper_Nesting.json'

class Longer_Chain(cofe):
    data_files = 'data/cofe/only_primitive_coverage/Longer_Chain.json'

class Phrase_Recombination(cofe):
    data_files = 'data/cofe/only_primitive_coverage/Phrase_Recombination.json'

class Primitive_Structural_Alternation(cofe):
    data_files = 'data/cofe/only_primitive_coverage/Primitive_Structural_Alternation.json'

class Primitive_Substitution(cofe):
    data_files = 'data/cofe/only_primitive_coverage/Primitive_Substitution.json'

class PR_LC(cofe):
    data_files = 'data/cofe/only_primitive_coverage/PR_LC.json'

class PR_LC_no_limit(cofe):
    data_files = 'data/cofe/only_primitive_coverage/PR_LC_no_limit_context.json'

class PR_LC_compose_incontext(cofe):
    data_files = 'data/cofe/only_primitive_coverage/PR_LC_compose_incontext.json'

class passive_to_active(cofe):
    data_files = 'data/cofe/pas_act_obj_subj/passive_to_active.json'

class obj_to_subj(cofe):
    data_files = 'data/cofe/pas_act_obj_subj/obj_to_subj.json'

class compose_passive_to_active_obj_to_subj(cofe):
    data_files = 'data/cofe/pas_act_obj_subj/compose_pa_os.json'

class compose_passive_to_active_obj_to_subj_compose_incontext(cofe):
    data_files = 'data/cofe/pas_act_obj_subj/pa_os_compose_incontext.json'
