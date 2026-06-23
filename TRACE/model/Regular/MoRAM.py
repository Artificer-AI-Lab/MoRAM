import os
import torch
from tqdm import tqdm

from model.base_model import CL_Base_Model
from utils.model.model_utils import TIKTOK
from utils.utils import print_rank_0, to_device
from utils.my_peft.tuners.moram import MoRAMLinear


class MoRAM(CL_Base_Model):
    """
    Continual learning trainer that applies the MoRAM (Mixture-of-Ranks) adapter framework.
    Built on top of the TreeLoRA infrastructure but without the orthogonal loss used by O-LoRA.
    """

    def __init__(self,
                 model,
                 tokenizer,
                 optimizer,
                 train_task_list,
                 eval_task_list,
                 test_task_list,
                 args):
        super().__init__(model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args)
        self.tiktok = TIKTOK(args)
        self.router_temperature = getattr(args, "moram_router_temp", 0.01)
        self.moram_infer_lora_a_thresh = getattr(args, "moram_infer_lora_a_thresh", 0.0)

        if self.args.local_rank == -1:
            self.device = torch.device("cuda")
        else:
            torch.cuda.set_device(self.args.local_rank)
            self.device = torch.device("cuda", self.args.local_rank)

        # Ensure router temperature propagated (in case not already set during model construction)
        for module in self.model.modules():
            if isinstance(module, MoRAMLinear):
                module.set_router_temperature(self.router_temperature)
                module.set_moram_infer_lora_a_threshold(self.moram_infer_lora_a_thresh)

    def train_one_task(self, task, i_task, epochs):
        train_dataloader = self.train_task_list[task]

        total_steps = epochs * len(train_dataloader)
        progress_bar = tqdm(total=total_steps, leave=True, disable=(self.args.global_rank != 0))

        for epoch in range(epochs):
            self.tiktok = TIKTOK(self.args)
            print_rank_0(
                f"Beginning of Epoch {epoch + 1}/{epochs}, Total Micro Batches {len(train_dataloader)}",
                self.args.global_rank)
            self.model.train()
            self.tiktok.print_time()

            tmp_rounds = -1

            for step, batch in enumerate(train_dataloader):
                tmp_rounds += 1
                if 'sources' in batch:
                    del batch['sources']
                batch = to_device(batch, self.device)

                outputs = self.model(**batch, use_cache=False)
                loss = outputs.loss

                if self.args.global_rank == 0:
                    progress_bar.update(1)
                    progress_bar.set_description(f"Epoch {epoch + 1}, Step {step}, Loss: {loss.item():.4f}",
                                                 refresh=False)

                self.tiktok.tik()
                self.model.backward(loss)
                self.model.step()
                self.tiktok.tok('backward time')

                if self.args.global_rank == 0 and tmp_rounds % 30 == 0:
                    print_rank_0(f"step={step}, loss={loss.item():.4f}", self.args.global_rank)
                    self.tiktok.print_time()

    def save_model(self, round):
        """
        Merge trained loranew into lora_A/B before writing checkpoints so adapter files never
        use rank-0 lora (empty accumulators). Fresh loranew is re-init inside increment_task.
        """
        for module in self.model.modules():
            if isinstance(module, MoRAMLinear):
                module.increment_task()

        if self.args.output_dir is not None:
            print_rank_0('saving the final model ...', self.args.global_rank)

        if self.args.global_rank == 0:
            peft_model_id = os.path.join(self.args.output_dir, str(round))
            if not os.path.exists(peft_model_id):
                os.makedirs(peft_model_id)
            self.model.save_pretrained(peft_model_id)
            self.tokenizer.save_pretrained(peft_model_id)
            print_rank_0(f'Successfully saving the final model to {peft_model_id}', self.args.global_rank)
