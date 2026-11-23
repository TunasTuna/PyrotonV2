"""Completion-masked chosen NLL for decoder-only TRL 0.29.0 ORPO."""

from torch.nn import functional as F
from trl.experimental.orpo import ORPOTrainer


class CompletionOnlyORPOTrainer(ORPOTrainer):
    def concatenated_forward(self, model, batch):
        if self.is_encoder_decoder:
            raise ValueError("Completion-only Pyroton ORPO requires a decoder-only model.")

        combined = self.concatenated_inputs(
            batch, is_encoder_decoder=False, padding_value=self.padding_value, device=self.accelerator.device
        )
        options = {"output_router_logits": True} if self.aux_loss_enabled else {}
        outputs = model(
            combined["concatenated_input_ids"],
            attention_mask=combined["concatenated_attention_mask"],
            use_cache=False,
            **options,
        )
        logits = outputs.logits
        labels = combined["concatenated_labels"]
        chosen_count = batch["chosen_labels"].shape[0]
        shifted_logits = logits[:, :-1, :]
        chosen_logits = shifted_logits[:chosen_count]
        chosen_labels = labels[:chosen_count, 1:]
        chosen_nll = F.cross_entropy(
            chosen_logits.reshape(-1, logits.shape[-1]),
            chosen_labels.reshape(-1).to(logits.device),
            ignore_index=-100,
        )
        logps = self.get_batch_logps(logits, labels, average_log_prob=True, is_encoder_decoder=False)
        result = (logps[:chosen_count], logps[chosen_count:], chosen_logits, shifted_logits[chosen_count:], chosen_nll)
        return (*result, outputs.aux_loss) if self.aux_loss_enabled else result
