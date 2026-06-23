import logging
import torch
from transformers.data.data_collator import *
from inference.ICL import TASK_PROMT, Constrained_PROMPT, QWEN3_MCQ_SUFFIX, QWEN3_MATH_SUFFIX, MCQ_TASKS, MATH_TASKS

logger = logging.getLogger(__name__)

# Must match TASK_PROMT["MeetingBank"] prefix in inference/ICL.py (long transcripts).
_MEETINGBANK_PROMPT_PREFIX = "Write a summary of the following meeting transcripts"


@dataclass
class DataCollator:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True  # ‘longest’
    max_prompt_len: Optional[int] = None
    max_ans_len: Optional[int] = None
    pad_to_multiple_of: Optional[int] = 1
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    inference: bool = False
    demonstrations: Optional[Any] = None
    task: str = None
    model_name: str = None  # For model-specific prompt formatting (e.g., Qwen3)
    # Long-context cap for MeetingBank only (set via CLI; None = use max_prompt_len everywhere).
    meetingbank_max_prompt_len: Optional[int] = None

    def _effective_max_prompt(self, instruction: str) -> int:
        if self.meetingbank_max_prompt_len is None:
            return self.max_prompt_len
        is_meetingbank = self.task == "MeetingBank" or instruction.startswith(
            _MEETINGBANK_PROMPT_PREFIX
        )
        if is_meetingbank:
            return self.meetingbank_max_prompt_len
        return self.max_prompt_len

    def _is_meetingbank(self, instruction: str) -> bool:
        return self.task == "MeetingBank" or instruction.startswith(
            _MEETINGBANK_PROMPT_PREFIX
        )

    def __call__(self, batch, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        model_inputs = self.decoder_call(batch, self.return_tensors)

        return model_inputs

    # only support left padding for now
    def tokenize(
        self,
        sentence,
        cutoff_len,
        add_bos_token=True,
        add_eos_token=True,
        keep_prompt_prefix=False,
    ):
        # Default truncation_side is "left" repo-wide (keep tail). For MeetingBank, long prompts
        # must keep the *start* (task text + transcript beginning); otherwise the model never
        # sees "Write a summary..." and ROUGE stays flat regardless of max length.
        prev_side = self.tokenizer.truncation_side
        if keep_prompt_prefix:
            self.tokenizer.truncation_side = "right"
        try:
            result = self.tokenizer(
                sentence,
                truncation=True,
                max_length=cutoff_len,
                add_special_tokens=False,
                padding=False,
                return_tensors=None,
            )
        finally:
            self.tokenizer.truncation_side = prev_side

        if (
                len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        if (
                len(result["input_ids"]) < cutoff_len
                and add_bos_token
        ):
            result["input_ids"] = [self.tokenizer.bos_token_id] + result["input_ids"]
            result["attention_mask"] = [1] + result["attention_mask"]

        result["labels"] = result["input_ids"].copy()

        return result

    def _tokenize_meetingbank_train(self, instruction, label, limit_len):
        """Instruction (transcript) truncated from the end; full label (summary) kept at sequence tail."""
        label_cut = min(limit_len, max(self.max_ans_len + 2, 32))
        tokenized_label = self.tokenize(
            label,
            label_cut,
            add_bos_token=False,
            add_eos_token=True,
            keep_prompt_prefix=False,
        )
        label_ids = list(tokenized_label["input_ids"])
        label_len = len(label_ids)
        bos_id = self.tokenizer.bos_token_id
        prefix_ids = [bos_id] if bos_id is not None else []
        instr_cap = limit_len - len(prefix_ids) - label_len
        if instr_cap < 1:
            instr_cap = 1
        prev_side = self.tokenizer.truncation_side
        self.tokenizer.truncation_side = "right"
        try:
            enc_i = self.tokenizer(
                instruction,
                truncation=True,
                max_length=instr_cap,
                add_special_tokens=False,
                padding=False,
                return_tensors=None,
            )
        finally:
            self.tokenizer.truncation_side = prev_side
        instr_ids = list(enc_i["input_ids"])
        input_ids = prefix_ids + instr_ids + label_ids
        if len(input_ids) > limit_len:
            over = len(input_ids) - limit_len
            instr_ids = instr_ids[: max(0, len(instr_ids) - over)]
            input_ids = prefix_ids + instr_ids + label_ids
        att = [1] * len(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": att,
            "labels": input_ids.copy(),
        }, label_len

    # support decoder-only models for left padding
    def decoder_call(self, batch, return_tensors):
        # to fix the bug
        sources = []
        gts = []
        tokenized_sources = []
        label_lens = []
        actual_max_len = 0

        for instance in batch:
            instruction = instance['prompt']
            label = instance['answer']
            sources.append(instruction)
            gts.append(label)

            max_prompt = self._effective_max_prompt(instruction)
            limit_len = max_prompt + self.max_ans_len if not self.inference else max_prompt

            if not self.inference:
                if self._is_meetingbank(instruction):
                    tokenize_source, lb_len = self._tokenize_meetingbank_train(
                        instruction, label, limit_len
                    )
                    label_lens.append(lb_len)
                    tokenized_sources.append(tokenize_source)
                else:
                    tokenized_label = self.tokenize(
                        label, limit_len, add_bos_token=False, add_eos_token=True
                    )
                    tokenize_source = self.tokenize(
                        instruction + label,
                        limit_len,
                        add_bos_token=True,
                        add_eos_token=True,
                    )
                    label_lens.append(len(tokenized_label["input_ids"]))
                    tokenized_sources.append(tokenize_source)
            else:
                if self.demonstrations!=None:
                    task_prompt = ""
                    task_prompt += TASK_PROMT[self.task]
                    if self.task!="MeetingBank":
                        task_prompt += Constrained_PROMPT
                    for demonstration in self.demonstrations:
                        if self.task=="Py150":
                            task_prompt+= "Code:\n"
                        task_prompt+=demonstration["prompt"]
                        task_prompt+=demonstration["answer"]+"\n\n"
                    
                    if self.task=="Py150":
                        task_prompt+= "Code:\n"
                    # task_prompt += Constrained_PROMPT
                    if self.task!="Py150":
                        instruction = instruction[len(TASK_PROMT[self.task]):]
                    instruction = task_prompt+instruction
                
                # Add Qwen3-specific standardized output format suffix
                if self.model_name and 'qwen3' in self.model_name.lower():
                    # if self.task in MCQ_TASKS:
                    #     instruction = instruction + QWEN3_MCQ_SUFFIX
                    if self.task in MATH_TASKS:
                        instruction = instruction + QWEN3_MATH_SUFFIX
                    
                    # Use chat template for Qwen3
                    messages = [{"role": "user", "content": instruction}]
                    instruction = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    # Tokenize without adding bos (chat template already adds it)
                    tokenize_source = self.tokenize(
                        instruction,
                        limit_len,
                        add_bos_token=False,
                        add_eos_token=False,
                        keep_prompt_prefix=self._is_meetingbank(instruction),
                    )
                else:
                    tokenize_source = self.tokenize(
                        instruction,
                        limit_len,
                        add_bos_token=True,
                        add_eos_token=False,
                        keep_prompt_prefix=self._is_meetingbank(instruction),
                    )
                tokenized_sources.append(tokenize_source)

            if len(tokenize_source["input_ids"]) > actual_max_len:
                actual_max_len = len(tokenize_source["input_ids"])

        actual_pad_len = (
                    (actual_max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of * self.pad_to_multiple_of)

        for idx in range(len(tokenized_sources)):
            pad_len = actual_pad_len - len(tokenized_sources[idx]["input_ids"])
            assert sum(tokenized_sources[idx]["attention_mask"]) == len(tokenized_sources[idx]["input_ids"])
            tokenized_sources[idx]["input_ids"] = [self.tokenizer.pad_token_id] * pad_len + tokenized_sources[idx][
                "input_ids"]

            tokenized_sources[idx]["attention_mask"] = [0] * pad_len + tokenized_sources[idx]["attention_mask"]

            if not self.inference:
                label_len = label_lens[idx]
                label_mask_len = actual_pad_len - label_len
                tokenized_sources[idx]["labels"] = [-100] * label_mask_len + tokenized_sources[idx]["labels"][
                                                                             -label_len:]
                assert len(tokenized_sources[idx]["input_ids"]) == len(tokenized_sources[idx]["attention_mask"]) == len(
                    tokenized_sources[idx]["labels"]) == actual_pad_len

        model_inputs = {'input_ids': torch.tensor([source["input_ids"] for source in tokenized_sources]),
                        'attention_mask': torch.tensor([source["attention_mask"] for source in tokenized_sources])}

        if not self.inference:
            model_inputs['labels'] = torch.tensor([source["labels"] for source in tokenized_sources])

        model_inputs['sources'] = sources
        if self.inference:
            model_inputs['gts'] = gts

        return model_inputs
