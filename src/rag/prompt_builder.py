import logging

logger = logging.getLogger(__name__)


class PromptBuilder:
    def __init__(self, template: str = "chatml"):
        if template not in ("chatml", "llama3"):
            raise ValueError("template must be 'chatml' or 'llama3'")
        self.template = template

    def build_prompt(
        self,
        system_prompt: str,
        user_query: str,
        context_chunks: list[dict],
    ) -> str:
        context_str = self._format_context(context_chunks)
        full_user = f"{context_str}\n\n{user_query}" if context_str else user_query

        if self.template == "llama3":
            return (
                f"<|begin_of_text|><|start_header_id|>system<|eot_id|>\n"
                f"{system_prompt}<|eot_id|>\n"
                f"<|start_header_id|>user<|eot_id|>\n"
                f"{full_user}<|eot_id|>\n"
                f"<|start_header_id|>assistant<|eot_id|>\n"
            )
        else:
            return (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{full_user}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

    def _format_context(self, chunks: list[dict]) -> str:
        if not chunks:
            return ""
        parts = []
        for i, c in enumerate(chunks, 1):
            parts.append(f"[{i}] {c['text']}")
        return "Context:\n" + "\n".join(parts)
