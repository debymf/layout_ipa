import json
import os
import random

import numpy as np
import torch
from dynaconf import settings
from loguru import logger
from overrides import overrides
from prefect import Task
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    AutoModel,
)

from layout_ipa.models import LayoutlmForSequenceClassification,  LayoutlmConfig
from transformers import (
    AdamW,
    BertConfig,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)

LAYOUT_LM_MODEL = settings["layout_lm_base"]


class SelectionLayoutLMTrainer(Task):
    def __init__(self, **kwargs):
        super(SelectionLayoutLMTrainer, self).__init__(**kwargs)
        self.per_gpu_batch_size = kwargs.get("per_gpu_batch_size", 16)
        self.cuda = kwargs.get("cuda", True)
        self.gradient_accumulation_steps = kwargs.get("gradient_accumulation_steps", 1)
        self.num_train_epochs = kwargs.get("num_train_epochs", 20)
        self.learning_rate = kwargs.get("learning_rate", 1e-5)
        self.weight_decay = kwargs.get("weight_decay", 0.0)
        self.adam_epsilon = kwargs.get("adam_epsilon", 1e-8)
        self.warmup_steps = kwargs.get("warmup_steps", 0)
        self.max_grad_norm = kwargs.get("max_grad_norm", 1.0)
        self.logging_steps = kwargs.get("logging_steps", 5)
        self.args = kwargs

    def set_seed(self, n_gpu, seed=1500):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(seed)

    @overrides
    def run(
        self,
        train_dataset,
        dev_dataset,
        test_dataset,
        task_name,
        output_dir,
        bert_model="bert-base-uncased",
        num_labels=261,
        mode="train",
        eval_fn=None,
        save_optimizer=False,
        eval_params={},
    ):
        torch.cuda.empty_cache()
        device = torch.device(
            "cuda" if torch.cuda.is_available() and self.cuda else "cpu"
        )

        n_gpu = torch.cuda.device_count()
        self.logger.info(f"GPUs used {n_gpu}")
        train_batch_size = self.per_gpu_batch_size * max(1, n_gpu)

        train_dataloader = DataLoader(
            train_dataset, batch_size=train_batch_size, shuffle=True
        )
        dev_dataloader = DataLoader(
            dev_dataset, batch_size=train_batch_size, shuffle=False
        )

        self.set_seed(n_gpu)
        outputs = {}
        if mode == "train":
            bert_config =  LayoutlmConfig.from_pretrained(LAYOUT_LM_MODEL, num_labels=num_labels)
            model = LayoutlmForSequenceClassification.from_pretrained(
                LAYOUT_LM_MODEL, config=bert_config
            )
            
            model = model.to(device)
            if n_gpu > 1:
                model = torch.nn.DataParallel(model)
            epoch_results = self.train(
                model,
                train_dataloader,
                dev_dataloader,
                dev_dataset,
                device,
                n_gpu,
                eval_fn,
                f"{output_dir}/{task_name}",
                save_optimizer,
                eval_params,
            )
            outputs["epoch_results "] = epoch_results
        logger.info("Running evalutaion mode")
        logger.info(f"Loading from {output_dir}/{task_name}")

        bert_config =  LayoutlmConfig.from_pretrained(f"{output_dir}/{task_name}", num_labels=num_labels)
        model = LayoutlmForSequenceClassification.from_pretrained(
                f"{output_dir}/{task_name}", config=bert_config
        )


        model.to(device)
        score = self.eval(
            model,
            dev_dataloader,
            dev_dataset,
            device,
            n_gpu,
            eval_fn,
            eval_params,
            mode="dev",
        )
        outputs["dev"] = {
            "score": score,
        }
        if test_dataset is not None:
            test_data_loader = DataLoader(
                test_dataset, batch_size=train_batch_size, shuffle=False
            )
            score = self.eval(
                model,
                test_data_loader,
                test_dataset,
                device,
                n_gpu,
                eval_fn,
                eval_params,
                mode="test",
            )
            outputs["test"] = {
                "score": score,
            }

        return outputs

    def train(
        self,
        model,
        train_dataloader,
        dev_dataloader,
        dev_dataset,
        device,
        n_gpu,
        eval_fn,
        output_dir,
        save_optimizer,
        eval_params,
    ):
        results = {}
        best_score = 0.0
        t_total = (
            len(train_dataloader)
            // self.gradient_accumulation_steps
            * self.num_train_epochs
        )

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters, lr=self.learning_rate, eps=self.adam_epsilon,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=t_total,
        )
        # criterion = torch.nn.CrossEntropyLoss()

        global_step = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        train_iterator = trange(
            epochs_trained, int(self.num_train_epochs), desc="Epoch",
        )
        for epoch in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                model.train()
                batch = tuple(t.to(device) for t in batch)
                inputs = {
                    "input_ids": batch[0],
                    "position_ids": batch[1],
                    "token_type_ids": batch[2],
                    "bbox": batch[3],
                    "labels": batch[4]
                }


                outputs = model(**inputs)

                loss, logits = outputs[:2]

                preds = logits.detach().cpu().numpy()
                preds = np.argmax(preds, axis=1)

                print("=====================================")
                print("*** PREDS ****")
                print(preds)
                print("\n\n")

                print("**** LABEL *****")
                print(inputs["labels"].detach().cpu().numpy())
                print("\n\n")

                print("**** SCORE ******")
                score = eval_fn(preds, inputs["labels"].detach().cpu().numpy())
                print(score)
                print("\n\n")
                print("=====================================")
                if n_gpu > 1:
                    loss = (
                        loss.mean()
                    )  # mean() to average on multi-gpu parallel training
                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.max_grad_norm
                    )

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    if self.logging_steps > 0 and global_step % self.logging_steps == 0:
                        loss_scalar = (tr_loss - logging_loss) / self.logging_steps
                        learning_rate_scalar = scheduler.get_lr()[0]
                        epoch_iterator.set_description(
                            f"Loss :{loss_scalar} LR: {learning_rate_scalar}"
                        )
                        logging_loss = tr_loss
            score = self.eval(
                model,
                dev_dataloader,
                dev_dataset,
                device,
                n_gpu,
                eval_fn,
                eval_params,
                mode="dev",
            )
            results[epoch] = score
            with torch.no_grad():
                if score > best_score:
                    logger.success(f"Storing the new model with score: {score}")
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)

                    torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
                    logger.info(f"Saving model checkpoint to {output_dir}")
                    if save_optimizer:
                        torch.save(
                            optimizer.state_dict(),
                            os.path.join(output_dir, "optimizer.pt"),
                        )
                        torch.save(
                            scheduler.state_dict(),
                            os.path.join(output_dir, "scheduler.pt"),
                        )
                        logger.info(
                            "Saving optimizer and scheduler states to %s", output_dir
                        )
                    best_score = score

        return results

    def eval(
        self, model, dataloader, dataset, device, n_gpu, eval_fn, eval_params, mode
    ):
        if n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "position_ids": batch[1],
                    "token_type_ids": batch[2],
                    "bbox": batch[3],
                    "labels": batch[4]
                }
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += outputs[0].mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0
                )


        eval_loss = eval_loss / nb_eval_steps

        score = None
        if eval_fn is not None:
            preds = np.argmax(preds, axis=1)
            print("\n\n")
            print("=================================")
            print("**** PREDS ****")
            print(preds)

            print("***OUT LABEL IDS***")
            print(out_label_ids)

            
            print("=================================")
            print("\n\n")
            score = eval_fn(preds, out_label_ids)
            # if mode == "test":

            #     out_preds = {
            #         "preds": np.argmax(preds, axis=1).tolist(),
            #         "gold": out_label_ids.tolist(),
            #     }
            #     with open("./cache/bin_preds_bert.json", "w") as fp:
            #         json.dump(out_preds, fp)

        logger.info(f"Score:{score}")
        return score
