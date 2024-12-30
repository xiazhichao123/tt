
import os, sys
from pathlib import Path
from dataclasses import fields
import shutil, logging

root_dir = Path(__file__).absolute().parent
sys.path.insert(0, str(root_dir))

os.environ['HF_HUB_CACHE'] = str(root_dir / 'cache')
os.environ['HF_HOME'] = str(root_dir / 'cache')


import torch
import transformers
from datasets import load_dataset
import evaluate
import numpy as np


from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, default_data_collator, DataCollatorWithPadding, Trainer
from transformers import HfArgumentParser, set_seed
from transformers.training_args import TrainingArguments as training_args


from single_alg.model_args import ModelArguments
from single_alg.train_args import DataTrainingArguments
from single_alg.utils import get_latest_run



logger = logging.getLogger(__name__)

class TrainingArguments(training_args):
    def print(self):
        logger.info(f'{__file__}.{self.__class__.__name__}:**************************************')
        for _filed in fields(self):
            value = self.__dict__[_filed.name]
            logger.info(f'{_filed.name}: {value}\t\tmetadata: {_filed.metadata}')



class Huggingface_nlp_cls():
    def __init__(self, **kwargs):
        self.train_datasets = kwargs.pop('train_datasets')
        self.valid_datasets = kwargs.pop('val_datasets')
        self.resume = kwargs.pop('resume')
        self.checkpoint_patten = kwargs.pop('checkpoint_patten')

        self.is_mutil_label = False


        parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
        model_args, data_args, training_args = parser.parse_dict(kwargs, allow_extra_keys=True)
        model_args.print(), data_args.print(), training_args.print()

        if training_args.should_log:
            # The default of training_args.log_level is passive, so we set log level at info here to have that default.
            transformers.utils.logging.set_verbosity_info()

        log_level = training_args.get_process_log_level()
        logger.setLevel(log_level)
        transformers.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

        # Log on each process the small summary:
        logger.warning(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
            + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
        )
        logger.info(f"Training/evaluation parameters {training_args}")

        # Detecting last checkpoint.
        last_checkpoint = None
        if self.resume:
            if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
                last_checkpoint = get_latest_run(training_args.output_dir, self.checkpoint_patten)
                if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
                    raise ValueError(
                        f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                        "Use --overwrite_output_dir to overcome."
                    )
                elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
                    logger.info(
                        f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                        "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                    )
        self.last_checkpoint = last_checkpoint if last_checkpoint else None

        # Set seed before initializing model.
        set_seed(training_args.seed)
        self.model_args, self.data_args, self.training_args = model_args, data_args, training_args
        self.model_args.model_name_or_path = self.last_checkpoint if self.last_checkpoint else self.model_args.model_name_or_path


    def load_datasets(self):
        # Initialize our dataset and prepare it for the 'image-classification' task.
        dataset = load_dataset(
            'csv',
            data_files={'train': self.train_datasets, 'validation': self.valid_datasets},
            cache_dir=self.model_args.cache_dir,
            token=self.model_args.token,
            streaming=False,
            trust_remote_code=True
        )
        return dataset

    def _label_map(self):
        # Prepare label mappings.
        # We'll include these in the model's config to get human readable labels in the Inference API.
        labels = sorted(self.dataset["train"].unique('label'))
        label2id, id2label = {}, {}
        for i, label in enumerate(labels):
            label2id[label] = i
            id2label[i] = label
        return label2id, id2label

    def _load_config(self):
        config = AutoConfig.from_pretrained(self.model_args.model_name_or_path,
                                            num_labels=len(self.label2id),
                                            finetuning_task="text-classification",
                                            cache_dir=self.model_args.cache_dir,
                                            # revision=self.model_args.model_revision,
                                            token=self.model_args.token,
                                            label2id=self.label2id,
                                            id2label=self.id2label
                                            )

        if self.is_mutil_label:
            config.problem_type = "multi_label_classification"
            logger.info("setting problem type to multi label classification")
        else:
            config.problem_type = "single_label_classification"
            logger.info("setting problem type to single label classification")
        return config

    def _load_processor(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_args.model_name_or_path,
                                                  cache_dir=self.model_args.cache_dir,
                                                  use_fast=self.model_args.use_fast_tokenizer,
                                                  revision=self.model_args.model_revision,
                                                  token=self.model_args.token
                                                  )
        self.max_seq_length = min(self.data_args.max_seq_length, tokenizer.model_max_length)
        if self.data_args.pad_to_max_length:
            self.padding = "max_length"
        else:
            # We will pad later, dynamically at batch creation, to the max sequence length in each batch
            self.padding = False
        return tokenizer


    def multi_labels_to_ids(self, labels):

        ids = [0.0] * len(self.label2id)  # BCELoss requires float as target type
        for label in labels:
            ids[int(label)] = 1.0
        return ids


    def preprocess_function(self, examples):
        result = self.processors(examples["text"], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        if self.is_mutil_label:
            result["label"] = [self.multi_labels_to_ids(l) for l in examples["label"]]
        else:
            result["label"] = [self.label2id[l] for l in examples["label"]]
        return result


    def _datasets_transormed(self):
        # Running the preprocessing pipeline on all the datasets
        with self.training_args.main_process_first(desc="dataset map pre-processing"):
            self.dataset = self.dataset.map(
                self.preprocess_function,
                batched=True,
                # load_from_cache_file=not self.data_args.overwrite_cache,
                # desc="Running tokenizer on dataset",
            )

    def collate_fn(self):
        if self.data_args.pad_to_max_length:
            data_collator = default_data_collator
        elif self.training_args.fp16:
            data_collator = DataCollatorWithPadding(self.processors, pad_to_multiple_of=8)
        else:
            data_collator = None
        return data_collator

    def _def_metric(self):
        metric = evaluate.load(str(root_dir / 'single_alg/accuracy.py'), cache_dir=self.model_args.cache_dir)
        logger.info("Using accuracy as classification score, you can use --metric_name to overwrite.")
        return metric

    def _compute_metrics(self, p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        if self.is_mutil_label:
            preds = np.array([np.where(p > 0, 1, 0) for p in preds])  # convert logits to multi-hot encoding
            # Micro F1 is commonly used in multi-label classification
            result = self.metric.compute(predictions=preds, references=p.label_ids, average="micro")
        else:
            preds = np.argmax(preds, axis=1)
            result = self.metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result


    def _load_model(self):
        model = AutoModelForSequenceClassification.from_pretrained(self.model_args.model_name_or_path,
                                                                   from_tf=bool
                                                                       (".ckpt" in self.model_args.model_name_or_path),
                                                                   config=self.config,
                                                                   cache_dir=self.model_args.cache_dir,
                                                                   revision=self.model_args.model_revision,
                                                                   token=self.model_args.token,
                                                                   ignore_mismatched_sizes=True
                                                                   )
        return model


    def _load_trainer(self):
        # Initialize our trainer
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.dataset["train"] if self.training_args.do_train else None,
            eval_dataset=self.dataset["validation"] if self.training_args.do_eval else None,
            compute_metrics=self._compute_metrics,
            tokenizer=self.processors,
            data_collator=self.collate_fn(),
        )
        return trainer

    def do_train(self):
        # Training
        if self.training_args.do_train:
            checkpoint = None
            if self.resume:
                if self.training_args.resume_from_checkpoint is not None:
                    checkpoint = self.training_args.resume_from_checkpoint
                elif self.last_checkpoint is not None:
                    checkpoint = self.last_checkpoint
            train_result = self.trainer.train(resume_from_checkpoint=checkpoint)
            self.trainer.save_model()
            self.trainer.log_metrics("train", train_result.metrics)
            self.trainer.save_metrics("train", train_result.metrics)
            self.trainer.save_state()

    def do_eval(self):
        # Evaluation
        if self.training_args.do_eval:
            metrics = self.trainer.evaluate()
            self.trainer.log_metrics("eval", metrics)
            self.trainer.save_metrics("eval", metrics)

    def do_predict(self):
        if self.training_args.do_predict:
            inputs = self.processors(self.test_file, padding=self.padding, max_length=self.max_seq_length, truncation=True, return_tensors="pt")
            inputs = inputs.to(self.model.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits.cpu().numpy()


            for p in logits:
                if self.is_mutil_label:
                    predicted_class_id = np.where(p > 0)[0].tolist()
                    print('restult: ', ','.join([self.model.config.id2label[pre_id]  for pre_id in predicted_class_id]))
            else:
                predicted_class_id = p.argmax().item()
                print('restult: ', self.model.config.id2label[predicted_class_id])


    def _transforms(self):
        logger.info(f'no transformes for aug.')
        return None, None


    def __call__(self):
        self.dataset = self.load_datasets()
        self.label2id, self.id2label = self._label_map()

        self.config = self._load_config()
        self.processors = self._load_processor()
        self._train_transforms, self._valid_transforms = self._transforms()
        self._datasets_transormed()

        self.metric = self._def_metric()
        self.model = self._load_model()
        self.trainer = self._load_trainer()

        self.do_train()
        self.do_eval()
        self.do_predict()


if __name__ == '__main__':
    import platform

    train_datasets = str(root_dir / 'datasets/20241223/train_all.csv')
    val_datasets = str(root_dir / 'datasets/20241223/test_all.csv')
    model_name = 'SamLow_eroberta-base-go_emotions'
    if "Windows" in platform.platform():
        model_name_or_path = str(root_dir / f'weights/{model_name}')
        output_dir = str(root_dir / 'output')
    else:
        model_name_or_path = f'/data/xzc/weights/{model_name}'
        output_dir = '/data/xzc/weights/huggingface/nlp_cls/output'

    do_train = True
    do_eval = True
    do_predict = False
    hf_cache = str(root_dir / 'cache')
    # output_dir = str(root_dir / 'output')
    resume = False
    ignore_mismatched_sizes = True
    remove_unused_columns = False
    logging_steps = 0.9999
    num_train_epochs = 100
    per_device_train_batch_size = 16
    save_steps = 200
    checkpoint_patten = 'checkpoint*'

    model_args = {
        'train_datasets': train_datasets,
        'val_datasets': val_datasets,
        'model_name_or_path': model_name_or_path,
        'do_train': do_train,
        'do_eval': do_eval,
        'do_predict': do_predict,
        'hf_cache': hf_cache,
        'output_dir': output_dir,
        'resume': resume,
        'ignore_mismatched_sizes': ignore_mismatched_sizes,
        'remove_unused_columns': remove_unused_columns,
        'logging_steps': logging_steps,
        'num_train_epochs': num_train_epochs,
        'per_device_train_batch_size': per_device_train_batch_size,
        'save_steps': save_steps,
        'checkpoint_patten': checkpoint_patten,
    }

    if Path(hf_cache).exists():
        shutil.rmtree(hf_cache)
        Path(hf_cache).mkdir(parents=True, exist_ok=True)
    model = Huggingface_nlp_cls(**model_args)
    model()
