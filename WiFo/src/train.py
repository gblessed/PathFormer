# coding=utf-8
import csv
import os
import torch
from torch.optim import AdamW, SGD, Adam
import random
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error
import math
import time
import collections


class TrainLoop:
    def __init__(self, args, writer, model, train_data, test_data, device, early_stop = 5):
        self.args = args
        self.writer = writer
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.device = device
        self.lr_anneal_steps = args.lr_anneal_steps
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        trainable_params = [p for p in self.model.parameters() if p.requires_grad==True]
        if len(trainable_params) == 0:
            raise ValueError("No trainable parameters found for the selected WiFo finetuning configuration.")
        self.opt = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                            self.opt, 
                            mode='min',      
                            factor=0.1,     
                            patience=5,     
                            min_lr=self.args.min_lr    
                        )
        self.log_interval = args.log_interval
        self.best_nmse_random = 1e9
        self.warmup_steps=5
        self.min_lr = args.min_lr
        self.best_nmse = 1e9
        self.early_stop = early_stop
        self.remaining_patience = early_stop
        
        self.mask_list = {'random':[0.85],'temporal':[0.5], 'fre':[0.5]}

    @staticmethod
    def nmse_to_db(nmse, eps=1e-12):
        return 10.0 * np.log10(max(float(nmse), eps))


    def Sample(self, test_data, step, mask_ratio, mask_strategy, seed=None, dataset='', index=0):
        with torch.no_grad():
            error_nmse = 0
            error_nmse_last = 0
            num=0
            # start time
            for _, batch in enumerate(test_data[index]):

                loss, _, pred, target, mask = self.model_forward(batch, self.model, mask_ratio, mask_strategy, seed=seed, data = dataset, mode='forward')


                dim1 = pred.shape[0]
                pred_mask = pred.squeeze(dim=2)  # [N,240,32]
                target_mask = target.squeeze(dim=2)


                y_pred = pred_mask[mask==1].reshape(-1,1).reshape(dim1,-1).detach().cpu().numpy()  # [Batch_size, 样本点数目]
                y_target = target_mask[mask==1].reshape(-1,1).reshape(dim1,-1).detach().cpu().numpy()

                error_nmse += np.sum(np.mean(np.abs(y_target - y_pred) ** 2, axis=1) / np.mean(np.abs(y_target) ** 2, axis=1))
                # print(f"pred shape: {self.model.unpatchify(pred).shape}")
                pred_last = self.model.unpatchify(pred).detach().cpu().numpy()[:, -1, :, 16:]
                target_last = self.model.unpatchify(target).detach().cpu().numpy()[:, -1, :, 16:]
                error_nmse_last += np.sum(
                    np.mean(np.abs(target_last - pred_last) ** 2, axis=(1, 2))
                    / np.mean(np.abs(target_last) ** 2, axis=(1, 2))
                )
                num += y_pred.shape[0]  # 本轮mask的个数: 1000*576*0.5

        nmse = error_nmse / num
        nmse_last = error_nmse_last / num

        return {
            'nmse': np.float32(nmse),
            'nmse_db': np.float32(self.nmse_to_db(nmse)),
            'last_nmse': np.float32(nmse_last),
            'last_nmse_db': np.float32(self.nmse_to_db(nmse_last)),
        }


    def Evaluation(self, test_data, epoch, seed=None):
        self.model.eval()

        nmse_list = []
        nmse_key_result = {}

        for index, dataset_name in enumerate(self.args.dataset.split('*')):

            nmse_key_result[dataset_name] = {}

            if self.args.mask_strategy_random != 'none':
                mask_list = self.mask_list_chosen(dataset_name)  # 自定义mask_list
                for s in mask_list:
                    for m in self.mask_list[s]:
                        metrics = self.Sample(test_data, epoch, mask_ratio=m, mask_strategy = s, seed=seed, dataset = dataset_name, index=index)
                        nmse_list.append(metrics['nmse'])
                        if s not in nmse_key_result[dataset_name]:
                            nmse_key_result[dataset_name][s] = {}
                        nmse_key_result[dataset_name][s][m] = metrics
                        

                        self.writer.add_scalar('Test_NMSE/{}-{}-{}'.format(dataset_name.split('_C')[0], s, m), metrics['nmse'], epoch)
                        self.writer.add_scalar('Test_NMSE_dB/{}-{}-{}'.format(dataset_name.split('_C')[0], s, m), metrics['nmse_db'], epoch)
                        self.writer.add_scalar('Test_Last_NMSE/{}-{}-{}'.format(dataset_name.split('_C')[0], s, m), metrics['last_nmse'], epoch)
                        self.writer.add_scalar('Test_Last_NMSE_dB/{}-{}-{}'.format(dataset_name.split('_C')[0], s, m), metrics['last_nmse_db'], epoch)

            else:
                s = self.args.mask_strategy
                m = self.args.mask_ratio
                metrics = self.Sample(test_data, epoch, mask_ratio=m, mask_strategy = s, seed=seed, dataset = dataset_name, index=index)
                nmse_list.append(metrics['nmse'])
                if s not in nmse_key_result[dataset_name]:
                    nmse_key_result[dataset_name][s] = {}
                nmse_key_result[dataset_name][s][m] = metrics


                self.writer.add_scalar('Test_NMSE/Stage-{}-{}-{}-{}'.format(self.args.stage, dataset_name.split('_C')[0], s, m), metrics['nmse'], epoch)
                self.writer.add_scalar('Test_NMSE_dB/Stage-{}-{}-{}-{}'.format(self.args.stage, dataset_name.split('_C')[0], s, m), metrics['nmse_db'], epoch)
                self.writer.add_scalar('Test_Last_NMSE/Stage-{}-{}-{}-{}'.format(self.args.stage, dataset_name.split('_C')[0], s, m), metrics['last_nmse'], epoch)
                self.writer.add_scalar('Test_Last_NMSE_dB/Stage-{}-{}-{}-{}'.format(self.args.stage, dataset_name.split('_C')[0], s, m), metrics['last_nmse_db'], epoch)

        self.print_eval_metrics(epoch, nmse_key_result)
        
        self.loss_test = np.mean(nmse_list)

        is_break = self.best_model_save(epoch, self.loss_test, nmse_key_result)
        return is_break  # 输出的是“save”

    def print_eval_metrics(self, epoch, nmse_key_result):
        for dataset_name, strategy_map in nmse_key_result.items():
            for mask_strategy, ratio_map in strategy_map.items():
                for mask_ratio, metrics in ratio_map.items():
                    print(
                        "Epoch {} eval dataset={} mask_strategy={} mask_ratio={} "
                        "val_nmse {:.6f} val_nmse_db {:.6f} last_nmse {:.6f} last_nmse_db {:.6f}".format(
                            epoch,
                            dataset_name,
                            mask_strategy,
                            mask_ratio,
                            metrics["nmse"],
                            metrics["nmse_db"],
                            metrics["last_nmse"],
                            metrics["last_nmse_db"],
                        )
                    )

    def write_results_csv(self, step, nmse_key_result):
        csv_path = os.path.join(self.args.model_path, "results_summary.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "stage",
                    "epoch",
                    "dataset",
                    "mask_strategy",
                    "mask_ratio",
                    "nmse",
                    "nmse_db",
                    "last_nmse",
                    "last_nmse_db",
                ],
            )
            writer.writeheader()
            for dataset_name, strategy_map in nmse_key_result.items():
                for mask_strategy, ratio_map in strategy_map.items():
                    for mask_ratio, metrics in ratio_map.items():
                        writer.writerow(
                            {
                                "stage": self.args.stage,
                                "epoch": step,
                                "dataset": dataset_name,
                                "mask_strategy": mask_strategy,
                                "mask_ratio": mask_ratio,
                                "nmse": metrics["nmse"],
                                "nmse_db": metrics["nmse_db"],
                                "last_nmse": metrics["last_nmse"],
                                "last_nmse_db": metrics["last_nmse_db"],
                            }
                        )
        print(f"Saved CSV summary to {csv_path}")

    def best_model_save(self, step, nmse, nmse_key_result):
        improved = nmse < self.best_nmse
        if improved:
            self.remaining_patience = self.early_stop
            torch.save(self.model.state_dict(), self.args.model_path+'model_save/model_best_stage_{}.pkl'.format(self.args.stage))
            torch.save(self.model.state_dict(), self.args.model_path+'model_save/model_best.pkl')
            self.best_nmse = nmse
            self.writer.add_scalar('Evaluation/NMSE_best', self.best_nmse, step)
            print('\nNMSE_best:{}\n'.format(self.best_nmse))
            print(str(nmse_key_result) + '\n')
            self.write_results_csv(step, nmse_key_result)
            with open(self.args.model_path+'result.txt', 'w') as f:
                f.write('stage:{}, epoch:{}, best nmse: {}\n'.format(self.args.stage, step, self.best_nmse))
                f.write(str(nmse_key_result) + '\n')
            with open(self.args.model_path+'result_all.txt', 'a') as f:
                f.write('stage:{}, epoch:{}, best nmse: {}\n'.format(self.args.stage, step, self.best_nmse))
                f.write(str(nmse_key_result) + '\n')
            return 'save'

        self.remaining_patience -= 1
        print('\nNMSE did not improve. patience left: {}\n'.format(self.remaining_patience))
        if self.remaining_patience <= 0:
            return 'break'
        return 'continue'

    def mask_select(self,name):
        if self.args.mask_strategy_random == 'none': #'none' or 'batch'
            mask_strategy = self.args.mask_strategy
            mask_ratio = self.args.mask_ratio
        else:
            mask_strategy = random.choice(['random','temporal','fre'])
            mask_ratio = random.choice(self.mask_list[mask_strategy])

        return mask_strategy, mask_ratio


    def mask_list_chosen(self,name):
        if self.args.mask_strategy_random == 'none': #'none' or 'batch'
            mask_list = self.mask_list
        else:
            mask_list = {key: self.mask_list[key] for key in ['random','temporal','fre']}
        return mask_list

    def run_loop(self):
        if not getattr(self.args, 'do_finetune', False):
            self.Evaluation(self.test_data, 0)
            return

        if self.train_data is None:
            raise ValueError("Finetuning requested but no train_data was provided.")

        self.Evaluation(self.test_data, 0)

        for epoch in range(1, self.args.finetune_epochs + 1):
            train_metrics = self.train_one_epoch(epoch)
            self.writer.add_scalar('Train/Loss', train_metrics['loss'], epoch)
            self.writer.add_scalar('Train/Masked_Loss', train_metrics['masked_loss'], epoch)
            self.writer.add_scalar('Train/Visible_Loss', train_metrics['visible_loss'], epoch)

            print(
                'Epoch {} train_loss {:.6f} masked_loss {:.6f} visible_loss {:.6f}'.format(
                    epoch,
                    train_metrics['loss'],
                    train_metrics['masked_loss'],
                    train_metrics['visible_loss'],
                )
            )

            if getattr(self.args, 'save_every_epoch', False):
                torch.save(self.model.state_dict(), self.args.model_path+'model_save/model_epoch_{}.pkl'.format(epoch))

            if epoch % self.args.eval_interval != 0:
                continue

            status = self.Evaluation(self.test_data, epoch)
            self.scheduler.step(self.loss_test)
            if status == 'break':
                print('Early stopping triggered at epoch {}'.format(epoch))
                break

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_loss_masked = 0.0
        total_loss_visible = 0.0
        num_steps = 0

        for index, dataset_name in enumerate(self.args.dataset.split('*')):
            for batch in self.train_data[index]:
                mask_strategy, mask_ratio = self.mask_select(dataset_name)
                loss, loss2, _, _, _ = self.model_forward(
                    batch,
                    self.model,
                    mask_ratio,
                    mask_strategy,
                    data=dataset_name,
                    mode='backward',
                )

                self.opt.zero_grad()
                loss.backward()
                if self.args.clip_grad is not None and self.args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
                self.opt.step()

                total_loss += float(loss.item())
                total_loss_masked += float(loss.item())
                total_loss_visible += float(loss2.item())
                num_steps += 1

        if num_steps == 0:
            raise RuntimeError('No training batches were produced during finetuning.')

        return {
            'loss': total_loss / num_steps,
            'masked_loss': total_loss_masked / num_steps,
            'visible_loss': total_loss_visible / num_steps,
        }

    def model_forward(self, batch, model, mask_ratio, mask_strategy, seed=None, data=None, mode='backward'):
        pathformer_features = None
        if isinstance(batch, (list, tuple)) and len(batch) == 2 and torch.is_tensor(batch[0]) and torch.is_tensor(batch[1]):
            channel_batch, pathformer_features = batch
            batch = [i.to(self.device) for i in channel_batch]
            pathformer_features = pathformer_features.to(self.device)
        else:
            batch = [i.to(self.device) for i in batch]

        loss, loss2, pred, target, mask = self.model(
            batch,
            mask_ratio=mask_ratio,
            mask_strategy=mask_strategy,
            seed=seed,
            data=data,
            pathformer_features=pathformer_features,
        )
        return loss, loss2, pred, target, mask




'''
python /home/blessedg/Pathformer/generate_wifo_sub6_to_mmwave_data.py \
  --scenario city_6_miami_3p5
'''
