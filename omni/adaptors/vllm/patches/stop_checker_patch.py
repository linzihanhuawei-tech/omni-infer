import os
from vllm.logger import init_logger
from vllm.v1.request import Request,RequestStatus

logger = init_logger(__name__)

QWEN_3_THINK_START_TOKEN_ID = 151667  # <think>
QWEN_3_THINK_END_TOKEN_ID = 151668  # </think>
QWEN_3_AFTER_THINK_END_TOKEN_ID1 = 271  # \n\n
QWEN_3_AFTER_THINK_END_TOKEN_ID2 = 9612  # \n

DEEPSEEK_R1_THINK_START_TOKEN_ID = 128798 # <think>
DEEPSEEK_R1_THINK_END_TOKEN_ID = 128799 # </think>
DEEPSEEK_R1_AFTER_THINK_END_TOKEN_ID1 = 201 # \n
DEEPSEEK_R1_AFTER_THINK_END_TOKEN_ID2 = 271 # \n\n

SLIDING_WINDOW_SIZE = -16

Request._cot_end_symbol = False

def is_cot_end(self: Request) -> bool:
    if self._cot_end_symbol:
        return True

    if len(self.token_ids) <= 1:
        return False

    cot_end = False
    check_token_ids = self.token_ids[SLIDING_WINDOW_SIZE:]

    # Qwen 3 output ends with "</think>\n\n" or "</think>\n"
    if QWEN_3_THINK_END_TOKEN_ID in check_token_ids:
        check_index = check_token_ids.index(QWEN_3_THINK_END_TOKEN_ID) + 1
        if check_index < len(check_token_ids) and check_token_ids[check_index] in [
            QWEN_3_AFTER_THINK_END_TOKEN_ID1, QWEN_3_AFTER_THINK_END_TOKEN_ID2
        ]:
            cot_end = True

    # Deepseek output ends with "</think>\n" or "</think>\n\n"
    if DEEPSEEK_R1_THINK_END_TOKEN_ID in check_token_ids:
        check_index = check_token_ids.index(DEEPSEEK_R1_THINK_END_TOKEN_ID) + 1
        if check_index < len(check_token_ids) and check_token_ids[check_index] in [
            DEEPSEEK_R1_AFTER_THINK_END_TOKEN_ID1, DEEPSEEK_R1_AFTER_THINK_END_TOKEN_ID2
        ]:
            cot_end = True


    if cot_end:
        self._cot_end_symbol = True

    return self._cot_end_symbol

def check_stop(request: Request, max_model_len: int) -> bool:
    enable_max_tokens_exclude_reasoning = os.environ.get("ENABLE_MAX_TOKENS_EXCLUDE_REASONING","0") == "1"
    reasoing_max_tokens: int = int(os.getenv("REASONING_MAX_TOKENS","32768"))
    role = os.environ.get("ROLE")
    if role != 'prefill': # decode节点上生效
        if not request.is_cot_end() and request.num_output_tokens >= reasoing_max_tokens:
            request.status = RequestStatus.FINISHED_LENGTH_CAPPED
            return True
        if enable_max_tokens_exclude_reasoning and not request.is_cot_end():
            prompt_len = request.num_output_tokens
            # prompt + 输出的max_tokens不能超过max_model_len
            if prompt_len + request.max_tokens < max_model_len:
                request.max_tokens = request.max_tokens + 1

    if (request.num_tokens >= max_model_len or request.num_output_tokens >= request.max_tokens):
        request.status = RequestStatus.FINISHED_LENGTH_CAPPED
        return True

    sampling_params = request.sampling_params
    last_token_id = request.output_token_ids[-1]
    if (not sampling_params.ignore_eos and last_token_id == request.eos_token_id):
        request.status = RequestStatus.FINISHED_LENGTH_CAPPED
        return True

    if last_token_id in (sampling_params.stop_token_ids or ()):
        request.status = RequestStatus.FINISHED_LENGTH_CAPPED
        request.stop_reason = last_token_id
        return True
    return False

def patch_enable_max_token_exclude_reasoning():
    from vllm.v1.core.sched import utils
    from vllm.v1.core.sched import scheduler
    from vllm.v1 import request

    print('+++++++ patch_enable_max_token_exclude_reasoning ++++++++')
    request.Request.is_cot_end = is_cot_end
    utils.check_stop = check_stop
    scheduler.check_stop = check_stop