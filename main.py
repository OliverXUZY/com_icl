import argparse
import json
import logging
import os
import time
import pathlib

from lm_eval import tasks, evaluator, utils
from lm_eval.utils import time_str

logging.getLogger("openai").setLevel(logging.WARNING)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_args", default="")
    parser.add_argument(
        "--tasks", default=None, choices=utils.MultiChoice(tasks.ALL_TASKS)
    )
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=str, default=None)
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=None,
        help="Maximal batch size to try with --batch_size auto",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Limit the number of examples per task. "
        "If <1, limit is a percentage of the total number of examples.",
    )
    parser.add_argument("--data_sampling", type=float, default=None)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--decontamination_ngrams_path", default=None)
    parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")
    parser.add_argument("--write_out", action="store_true", default=False)
    parser.add_argument("--output_base_path", type=str, default=None)

    parser.add_argument("--rnd_seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()

    assert not args.provide_description  # not implemented

    if args.limit:
        print(
            "WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )

    if args.tasks is None:
        task_names = tasks.ALL_TASKS
    else:
        task_names = utils.pattern_match(args.tasks.split(","), tasks.ALL_TASKS)

    print(f"Selected Tasks: {task_names}")

    description_dict = {}
    if args.description_dict_path:
        with open(args.description_dict_path, "r") as f:
            description_dict = json.load(f)

    start = time.time()

    results = evaluator.simple_evaluate(
        model=args.model,
        model_args=args.model_args,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        device=args.device,
        no_cache=args.no_cache,
        limit=args.limit,
        description_dict=description_dict,
        decontamination_ngrams_path=args.decontamination_ngrams_path,
        check_integrity=args.check_integrity,
        write_out=args.write_out,
        output_base_path=args.output_base_path,
        rnd_seed = args.rnd_seed,
    )

    dumped = json.dumps(results, indent=2)
    print(dumped)

    if args.output_path:
        dirname = os.path.dirname(args.output_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(args.output_path, "w") as f:
            f.write(dumped)

    batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))
    print(
        f"{args.model} ({args.model_args}), limit: {args.limit}, provide_description: {args.provide_description}, "
        f"num_fewshot: {args.num_fewshot}, batch_size: {args.batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
    )
    print(evaluator.make_table(results))


    # added by zhuoyan, log value summary
    log_path = args.output_base_path

    if log_path:
        log_path = (
            pathlib.Path(log_path)
            if log_path is not None
            else pathlib.Path(".")
        )
        try:
            log_path.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            pass
    for task_name, task_dict in results["results"].items():
        values_summary = {}
        values_summary["task"] = task_name
        values_summary["rnd_seed"] = args.rnd_seed
        values_summary['prompt_des'] = True if args.description_dict_path else False
        values_summary['num_fewshot'] = args.num_fewshot
        values_summary['batch_size'] = f"{args.batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"

        
        # values_summary['tag'] = args.tag

        
        for m, v in task_dict.items():
            values_summary[m] = v

        
        
        with open(os.path.join(log_path, "log_summary.json"), 'a') as f:
            f.write(str(values_summary) + '\n')
    
    print("run rnd_seed:{} with fewshot {} [took {}]".format(
        args.rnd_seed,
        args.num_fewshot, 
        time_str(time.time() - start))
        )



if __name__ == "__main__":
    main()
