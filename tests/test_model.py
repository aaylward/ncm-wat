import torch

from ncm.config import make_config
from ncm.generator import Generator, code_loss, sample
from ncm.text import PAD
from ncm.vq import VQTokenizer

VOCAB = 50


def tiny():
    torch.manual_seed(0)
    cfg = make_config("tiny")
    vq = VQTokenizer(cfg.vq, VOCAB)
    return cfg, vq


def test_vq_shapes_and_eval_forward_matches_encode():
    cfg, vq = tiny()
    ids = torch.randint(0, VOCAB, (3, cfg.vq.block_len))
    vq.train()
    for _ in range(3):  # initializes codebooks from data and runs EMA updates
        logits, loss, codes = vq(ids)
    assert logits.shape == (3, cfg.vq.block_len, VOCAB)
    assert codes.shape == (3, sum(cfg.vq.levels)) and loss.item() > 0
    vq.eval()
    _, _, codes = vq(ids)
    assert torch.equal(codes, vq.encode(ids))
    assert torch.equal(vq.decode(codes), vq.decode_z(vq.vq.decode(codes)).argmax(-1))
    assert vq.decode_levels(codes)[-1].shape == (3, cfg.vq.block_len)


def test_vq_codebooks_update_only_in_training():
    cfg, vq = tiny()
    ids = torch.randint(0, VOCAB, (4, cfg.vq.block_len))
    vq.eval()
    before = vq.vq.codebook.clone()
    vq(ids)
    assert torch.equal(before, vq.vq.codebook)
    vq.train()
    vq(ids)
    assert not torch.equal(before, vq.vq.codebook)
    assert not vq.vq.codebook.requires_grad


def test_vq_gradients_reach_encoder_and_decoder():
    cfg, vq = tiny()
    ids = torch.randint(0, VOCAB, (4, cfg.vq.block_len))
    logits, loss, _ = vq(ids)
    (torch.nn.functional.cross_entropy(logits.reshape(-1, VOCAB), ids.reshape(-1)) + loss).backward()
    for p in (vq.wte.weight, vq.head.weight, vq.vq.down0.weight, vq.vq.phi[1].conv.weight):
        assert p.grad is not None and p.grad.abs().sum() > 0


def build_gen(cfg, vq):
    torch.manual_seed(1)
    return Generator(cfg.gen, cfg.vq, VOCAB, vq.vq.codebook).eval()


def batch(cfg, B=2):
    P = cfg.gen.prefix_len
    prefix = torch.randint(1, VOCAB, (B, P))
    prefix[0, : P // 2] = PAD  # left padding on one row
    return prefix


def test_cached_level_by_level_generation_equals_teacher_forced_forward():
    """The incremental path (prefix KV cache, one pass per level, inputs built level by level)
    must produce exactly the logits of the single teacher-forced pass used in training."""
    cfg, vq = tiny()
    vq.eval()
    gen = build_gen(cfg, vq)
    prefix = batch(cfg)
    codes = vq.encode(torch.randint(0, VOCAB, (2, cfg.vq.block_len)))
    full = gen(prefix, vq.vq.teacher_inputs(codes))
    out, inc = gen.generate(prefix, vq.vq, teacher_codes=codes, return_logits=True)
    assert torch.equal(out, codes)
    torch.testing.assert_close(inc, full, atol=1e-5, rtol=1e-4)


def test_level_logits_ignore_other_levels_and_pad_positions():
    cfg, vq = tiny()
    gen = build_gen(cfg, vq)
    prefix = batch(cfg)
    inputs = torch.randn(2, gen.T - cfg.vq.levels[0], cfg.vq.embed_dim)
    base = gen(prefix, inputs)
    # perturb level 2's inputs: only level 2's logits may change
    o2, l2 = gen.offsets[2], cfg.vq.levels[2]
    pert = inputs.clone()
    pert[:, o2 - cfg.vq.levels[0]: o2 - cfg.vq.levels[0] + l2] += 1.0
    out = gen(prefix, pert)
    changed = (out - base).abs().amax(dim=(0, 2)) > 1e-6
    assert changed[o2:o2 + l2].all() and not changed[:o2].any() and not changed[o2 + l2:].any()


def test_generate_samples_valid_codes_and_loss_is_finite():
    cfg, vq = tiny()
    vq.eval()
    gen = build_gen(cfg, vq)
    prefix = batch(cfg)
    codes = gen.generate(prefix, vq.vq, temperature=1.0, top_k=5, top_p=0.9)
    assert codes.shape == (2, gen.T) and codes.max() < cfg.vq.codebook_size
    assert torch.isfinite(code_loss(gen(prefix, vq.vq.teacher_inputs(codes)), codes))


def test_sample_greedy_and_top_k():
    logits = torch.tensor([[0.0, 5.0, 1.0]])
    assert sample(logits, temperature=0).item() == 1
    torch.manual_seed(0)
    assert all(sample(logits, temperature=1.0, top_k=1).item() == 1 for _ in range(10))
