from __future__ import annotations

from typing import TYPE_CHECKING, List, NamedTuple, NoReturn, Set, Tuple, TypeAlias

import torch
from minisgl.core import Batch, Req
from minisgl.env import ENV
from minisgl.message import (
    AbortBackendMsg,
    BaseBackendMsg,
    BatchBackendMsg,
    DetokenizeMsg,
    ExitMsg,
    UserMsg,
)
from minisgl.utils import init_logger, load_tokenizer

from .cache import CacheManager
from .config import SchedulerConfig
from .decode import DecodeManager
from .io import SchedulerIOMixin
from .prefill import ChunkedReq, PrefillManager
from .table import TableManager

if TYPE_CHECKING:
    from minisgl.engine import BatchSamplingArgs, ForwardOutput


logger = init_logger(__name__)

Indice2D: TypeAlias = Tuple[torch.Tensor, torch.Tensor]


# For overlap scheduling, we also need to cache some other data to avoid IMA
class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs
    input_tuple: Indice2D  # (token_mapping, positions)
    write_tuple: Indice2D  # (req_mapping, seq_lens or 0)


ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"


class Scheduler(SchedulerIOMixin):
    def __init__(self, config: SchedulerConfig):
        from minisgl.engine import Engine

        self.engine = Engine(config)

        # use another stream to overlap metadata processing with computation
        self.device = self.engine.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        torch.cuda.set_stream(self.stream)

        # initialize other managers
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        self.cache_manager = CacheManager(
            self.engine.num_pages, config.page_size, self.engine.page_table, config.cache_type
        )
        # Link tiered pool (if enabled) to cache manager for location tracking
        from minisgl.kvcache.tiered_pool import TieredKVCachePool

        if isinstance(self.engine.kv_cache, TieredKVCachePool):
            self.cache_manager.set_tiered_pool(self.engine.kv_cache)

        self.decode_manager = DecodeManager(config.page_size)
        self.prefill_manager = PrefillManager(
            self.cache_manager, self.table_manager, self.decode_manager
        )

        # some alias for easy access
        self.finished_reqs: Set[Req] = set()
        self.tokenizer = load_tokenizer(config.model_path)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.token_pool = self.table_manager.token_pool
        self.prefill_budget = config.max_extend_tokens
        # self.config = config

        # Counter for periodic KV cache debug stats
        self._step_count = 0
        self._kv_stats_interval = 1  # log every N forward steps

        # Initialize the I/O mixin
        super().__init__(config, self.engine.tp_cpu_group)

    def run_when_idle(self) -> None:
        """Called when the scheduler is idle to perform background tasks."""
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        The main loop of overlapping scheduling and execution.

        It will overlap the execution of current batch and processing of last batch's results,
        which can effectively hide CPU latency and improve GPU utilization.
        """
        blocking = not (
            last_data is not None  # don't block if we have a batch to be processed
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            with self.engine_stream_ctx:  # run the batch in the engine's stream
                self.engine.stream.wait_stream(self.stream)
                ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(last_data)
        return ongoing_data

    def normal_loop(self) -> None:
        blocking = not (self.prefill_manager.runnable or self.decode_manager.runnable)
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(ongoing_data)

    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        if ENV.DISABLE_OVERLAP_SCHEDULING:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self.normal_loop()
        else:
            assert torch.cuda.current_stream() == self.stream
            data = None
            while True:
                data = self.overlap_loop(data)

    def shutdown(self) -> None:
        torch.cuda.synchronize(self.device)
        self.sync_all_ranks()
        self.engine.shutdown()

    def _process_last_data(self, last_data: ForwardData | None) -> None:
        if last_data is None:
            return

        batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
        copy_done.synchronize()
        reply: List[DetokenizeMsg] = []
        new_finished_reqs: Set[Req] = set()
        with self.cache_manager.lazy_free_region():
            for i, req in enumerate(batch.reqs):
                if isinstance(req, ChunkedReq):
                    continue
                next_token = next_tokens_cpu[i]
                req.append_host(next_token.unsqueeze(0))
                next_token = int(next_token.item())
                finished = not req.can_decode
                if not req.sampling_params.ignore_eos:
                    finished |= next_token == self.eos_token_id
                reply.append(DetokenizeMsg(uid=req.uid, next_token=next_token, finished=finished))

                # NOTE: overlap scheduling may make the request freed twice, skip second free
                if finished and req not in self.finished_reqs:
                    self.decode_manager.remove_req(req)
                    self._free_req_resources(req)
                    new_finished_reqs.add(req)
                elif batch.is_prefill:  # for prefill, non-chunk req, cache the prefix
                    self.cache_manager.cache_req(req, finished=False)

        self.finished_reqs = new_finished_reqs
        self.send_result(reply)

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        if isinstance(msg, BatchBackendMsg):
            for msg in msg.data:
                self._process_one_msg(msg)
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            logger.debug_rank0("Received user msg: %s", msg)
            input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
            max_output_len = max_seq_len - input_len
            if max_output_len <= 0:
                return logger.warning_rank0(
                    f"Input sequence length {input_len} exceeds {max_seq_len}, "
                    f"request {msg.uid} is dropped."
                )
            if msg.sampling_params.max_tokens > max_output_len:
                msg.sampling_params.max_tokens = max_output_len
                logger.warning_rank0(
                    f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
                )
            self.prefill_manager.add_one_req(msg)
        elif isinstance(msg, AbortBackendMsg):
            logger.debug_rank0("Aborting request %d", msg.uid)
            req_to_free = self.prefill_manager.abort_req(msg.uid)
            req_to_free = req_to_free or self.decode_manager.abort_req(msg.uid)
            if req_to_free is not None:
                self._free_req_resources(req_to_free)
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    def _free_req_resources(self, req: Req) -> None:
        self.table_manager.free(req.table_idx)
        self.cache_manager.cache_req(req, finished=True)

    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        self.engine.graph_runner.pad_batch(batch)
        self.cache_manager.allocate_paged(batch.reqs)

        # Tiered KV cache: ensure all prefix-cached pages are on GPU and
        # page_table entries hold valid GPU buffer offsets before metadata
        # is built for the attention kernel.
        self.cache_manager.ensure_batch_on_gpu(batch.reqs)

        # Check for multimodal requests
        batch.has_multimodal = any(
            getattr(r, "is_multimodal", False) for r in batch.reqs
        )

        # Aggregate multimodal data for prefill
        if batch.has_multimodal and batch.is_prefill:
            pv_list = [r.pixel_values for r in batch.reqs if r.pixel_values is not None]
            if pv_list:
                batch.pixel_values = torch.cat(pv_list, dim=0)
            gt_list = [r.image_grid_thw for r in batch.reqs if r.image_grid_thw is not None]
            if gt_list:
                batch.image_grid_thw = torch.cat(gt_list, dim=0)

        batch.positions = _make_positions(batch, self.device)
        input_mapping = _make_input_tuple(batch, self.device)
        write_mapping = _make_write_tuple(batch, self.device)
        batch.out_loc = self.engine.page_table[input_mapping]
        self.engine.attn_backend.prepare_metadata(batch)
        return ForwardInput(
            batch=batch,
            sample_args=self.engine.sampler.prepare(batch),
            input_tuple=input_mapping,
            write_tuple=write_mapping,
        )

    def _schedule_next_batch(self) -> ForwardInput | None:
        # TODO: support other policies: e.g. DECODE first
        batch = (
            self.prefill_manager.schedule_next_batch(self.prefill_budget)
            or self.decode_manager.schedule_next_batch()
        )
        if batch is not None:
            self._step_count += 1
            if self._step_count % self._kv_stats_interval == 0:
                self._log_kv_stats(batch)
        return self._prepare_batch(batch) if batch else None

    def _log_kv_stats(self, batch: Batch) -> None:
        from minisgl.kvcache.tiered_pool import TieredKVCachePool

        kv = self.engine.kv_cache
        free_pages = len(self.cache_manager.free_slots)
        cache_info = self.cache_manager.prefix_cache.size_info
        msg = (
            f"[KV-step {self._step_count}] "
            f"batch={batch.phase}(n={len(batch.reqs)}) "
            f"free_pages={free_pages} "
            f"prefix_cache(evict={cache_info.evictable_size} "
            f"prot={cache_info.protected_size} "
            f"total={cache_info.total_size})"
        )
        if isinstance(kv, TieredKVCachePool):
            msg += f" tiered=[{kv.debug_stats()}]"
        logger.info_rank0(msg)

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        batch, sample_args, input_mapping, output_mapping = forward_input
        batch.input_ids = self.token_pool[input_mapping]
        forward_output = self.engine.forward_batch(batch, sample_args)
        self.token_pool[output_mapping] = forward_output.next_tokens_gpu
        self.decode_manager.filter_reqs(forward_input.batch.reqs)
        return forward_output


def _make_positions(batch: Batch, device: torch.device) -> torch.Tensor:
    has_mrope = getattr(batch, "has_multimodal", False)

    if has_mrope and batch.is_prefill:
        # MRoPE prefill: concatenate mrope_positions from each request
        pos_list = []
        for req in batch.padded_reqs:
            if req.mrope_positions is not None:
                # mrope_positions is (3, req_seq_len), slice to extend_len
                pos = req.mrope_positions[:, req.cached_len : req.device_len]
                pos_list.append(pos)
            else:
                # Fallback: text-only req in a mixed batch, replicate 1D as 3D
                length = req.extend_len
                pos_1d = torch.arange(req.cached_len, req.device_len, dtype=torch.int32)
                pos = pos_1d.unsqueeze(0).expand(3, -1)
                pos_list.append(pos)
        positions = torch.cat(pos_list, dim=1).to(torch.int32)
        return positions.to(device, non_blocking=True)

    if has_mrope and batch.is_decode:
        # MRoPE decode: each token gets 3 identical positions based on delta
        positions_list = []
        for req in batch.padded_reqs:
            if req.mrope_position_delta is not None:
                pos_val = req.device_len - 1 + req.mrope_position_delta
            else:
                pos_val = req.cached_len
            pos = torch.full((3, 1), pos_val, dtype=torch.int32)
            positions_list.append(pos)
        positions = torch.cat(positions_list, dim=1)
        return positions.to(device, non_blocking=True)

    # Original 1D positions for text-only batches
    needed_size = sum(r.extend_len for r in batch.padded_reqs)
    indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        torch.arange(
            req.cached_len,
            req.device_len,
            dtype=torch.int32,
            out=indices_host[offset : offset + length],
        )
        offset += length
    return indices_host.to(device, non_blocking=True)


def _make_input_tuple(batch: Batch, device: torch.device) -> Indice2D:
    # positions can be (seq_len,) for 1D or (3, seq_len) for MRoPE
    num_tokens = batch.positions.shape[-1]
    mapping_host = torch.empty(num_tokens, dtype=torch.int64, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        mapping_host[offset : offset + length].fill_(req.table_idx)
        offset += length
    mapping_device = mapping_host.to(device, non_blocking=True)

    # For token_pool indexing, always use 1D sequential positions.
    # MRoPE positions (3, N) are for rotary embeddings, not token storage.
    if batch.positions.dim() > 1:
        pos_host = torch.empty(num_tokens, dtype=torch.int64, pin_memory=True)
        offset = 0
        for req in batch.padded_reqs:
            length = req.extend_len
            torch.arange(
                req.cached_len, req.device_len, dtype=torch.int64,
                out=pos_host[offset : offset + length],
            )
            offset += length
        positions = pos_host.to(device, non_blocking=True)
    else:
        positions = batch.positions.to(torch.int64)

    return mapping_device, positions


def _make_write_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_list = [req.table_idx for req in batch.reqs]
    mapping_host = torch.tensor(mapping_list, dtype=torch.int64, pin_memory=True)
    write_list = [(req.device_len if req.can_decode else -1) for req in batch.reqs]
    write_host = torch.tensor(write_list, dtype=torch.int64, pin_memory=True)
    return mapping_host.to(device, non_blocking=True), write_host.to(device, non_blocking=True)
