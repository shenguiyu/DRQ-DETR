import datetime
import json
import time

import torch

from ..misc import dist_utils, get_weight_size, stats
from ..optim.lr_scheduler import FlatCosineLRScheduler
from ._solver import BaseSolver
from .det_engine import evaluate, train_one_epoch


RED, ORANGE, RESET = "\033[91m", "\033[38;5;208m", "\033[0m"
coco_name_list = ["ap", "ap50", "ap75", "aps", "apm", "apl", "ar", "ar50", "ar75", "ars", "arm", "arl"]


class DetSolver(BaseSolver):
    def fit(self, cfg_str):
        self.train()
        args = self.cfg

        if dist_utils.is_main_process():
            with open(self.output_dir / "args.json", "w") as json_file:
                json_file.write(cfg_str)

        _, model_stats = stats(self.cfg)
        print(model_stats)
        print("-" * 42 + "Start training" + "-" * 43)

        self.self_lr_scheduler = False
        if args.lrsheduler is not None:
            iter_per_epoch = len(self.train_dataloader)
            print("     ## Using Self-defined Scheduler-{} ## ".format(args.lrsheduler))
            self.lr_scheduler = FlatCosineLRScheduler(
                self.optimizer,
                args.lr_gamma,
                iter_per_epoch,
                total_epochs=args.epoches,
                warmup_iter=args.warmup_iter,
                flat_epochs=args.flat_epoch,
                no_aug_epochs=args.no_aug_epoch,
                lr_scyedule_save_path=self.output_dir,
            )
            self.self_lr_scheduler = True

        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"number of trainable parameters: {n_parameters}")

        best_stat = {"epoch": -1}
        if self.last_epoch > 0:
            module = self.ema.module if self.ema else self.model
            test_stats, _ = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
                yolo_metrice=self.cfg.yolo_metrice,
            )
            for k in test_stats:
                best_stat["epoch"] = self.last_epoch
                best_stat[k] = test_stats[k][0]
                print(f"best_stat: {best_stat}")

        start_time = time.time()
        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, args.epoches):
            self.train_dataloader.set_epoch(epoch)
            self.criterion.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            if epoch == self.train_dataloader.collate_fn.stop_epoch:
                self.load_resume_state(str(self.output_dir / "best_stg1.pth"))
                self.ema.decay = self.train_dataloader.collate_fn.ema_restart_decay
                print(f"Refresh EMA at epoch {epoch} with decay {self.ema.decay}")

            train_stats = train_one_epoch(
                self.self_lr_scheduler,
                self.lr_scheduler,
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                max_norm=args.clip_max_norm,
                print_freq=args.print_freq,
                ema=self.ema,
                scaler=self.scaler,
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer,
                plot_train_batch_freq=args.plot_train_batch_freq,
                output_dir=self.output_dir,
                epoches=args.epoches,
                verbose_type=args.verbose_type,
            )

            if not self.self_lr_scheduler:
                if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                    self.lr_scheduler.step()

            self.last_epoch += 1

            if self.output_dir and epoch < self.train_dataloader.collate_fn.stop_epoch:
                checkpoint_paths = [self.output_dir / "last.pth"]
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f"checkpoint{epoch:04}.pth")
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
                yolo_metrice=self.cfg.yolo_metrice,
            )

            for k in test_stats:
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats[k]):
                        self.writer.add_scalar(f"Test/{k}_{coco_name_list[i]}", v, epoch)

                best_stat_tamp = best_stat.copy()
                if k in best_stat:
                    best_stat["epoch"] = epoch if test_stats[k][0] > best_stat[k] else best_stat["epoch"]
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat_tamp[k] = 0
                    best_stat["epoch"] = epoch
                    best_stat[k] = test_stats[k][0]

                print(f"best_stat: {best_stat}")

                if best_stat["epoch"] == epoch and self.output_dir:
                    print(RED + f"epoch:{best_stat_tamp['epoch']}->{best_stat['epoch']} ap:{best_stat_tamp[k]:.4f}->{best_stat[k]:.4f}")
                    if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                        dist_utils.save_on_master(self.state_dict(), self.output_dir / "best_stg2.pth")
                        print("save best_stg2.pth success.")
                    else:
                        dist_utils.save_on_master(self.state_dict(), self.output_dir / "best_stg1.pth")
                        print("save best_stg1.pth success.")
                    print(RESET, end="")
                elif epoch >= self.train_dataloader.collate_fn.stop_epoch:
                    self.ema.decay -= 0.0001
                    self.load_resume_state(str(self.output_dir / "best_stg1.pth"))
                    print(f"Refresh EMA at epoch {epoch} with decay {self.ema.decay}")

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                if coco_evaluator is not None:
                    (self.output_dir / "eval").mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ["latest.pth"]
                        if epoch % 50 == 0:
                            filenames.append(f"{epoch:03}.pth")
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval" / name)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"Training time {total_time_str}")

    def val(self):
        self.eval()
        module = self.ema.module if self.ema else self.model
        module.deploy()
        _, model_info = stats(self.cfg, module=module)
        print(ORANGE, "--------------------Model Info(fused)", model_info, "--------------------", RESET)
        get_weight_size(module)
        _, coco_evaluator = evaluate(
            module,
            self.criterion,
            self.postprocessor,
            self.val_dataloader,
            self.evaluator,
            self.device,
            True,
            self.output_dir,
            self.cfg.yolo_metrice,
        )

        if self.output_dir:
            dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")
