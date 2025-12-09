import torch
from transformers import T5Config, T5ForConditionalGeneration, LogitsProcessorList, StoppingCriteriaList
import re
import random

core = 'NNN'
span_len = 5
checkpoint = 'c2c_model.pt'
num_sample = 20


# ---------- Copied from your training script (minimal parts) ----------
LETTER_SET = set(list("ACDEFGHIKLMNPQRSTVWY"))

class CharTokenizer:
    def __init__(self):
        aa = list("ACDEFGHIKLMNPQRSTVWY")
        digits = list("0123456789")
        punct  = list(" <>/:-_=+.,;()[]{}\"'\\")
        letters = [chr(c) for c in range(65, 91)] + [chr(c) for c in range(97, 123)]
        basic = sorted(set(aa + digits + punct + letters + [" "]))
        self.pad_token = "<pad>"
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.unk_token = "<unk>"
        specials = [self.pad_token, self.bos_token, self.eos_token, self.unk_token]
        self.vocab = specials + basic
        self.stoi  = {t:i for i,t in enumerate(self.vocab)}
        self.itos  = {i:t for i,t in enumerate(self.vocab)}
        self.pad_token_id = self.stoi[self.pad_token]
        self.eos_token_id = self.stoi[self.eos_token]
        self.bos_token_id = self.stoi[self.bos_token]
        self.unk_token_id = self.stoi[self.unk_token]
    def encode(self, text, add_eos=False, max_length=None):
        ids = [self.stoi.get(ch, self.unk_token_id) for ch in text]
        if add_eos: ids.append(self.eos_token_id)
        if max_length is not None: ids = ids[:max_length]
        return ids
    def batch_decode(self, ids_batch, skip_special_tokens=True):
        outs = []
        data = ids_batch.tolist() if hasattr(ids_batch, "tolist") else ids_batch
        for ids in data:
            toks = [self.itos.get(int(i), self.unk_token) for i in ids]
            if skip_special_tokens:
                toks = [t for t in toks if t not in (self.pad_token, self.eos_token, self.bos_token, self.unk_token)]
            outs.append("".join(toks))
        return outs
    def convert_ids_to_tokens(self, ids):
        return [self.itos.get(int(i), self.unk_token) for i in ids]
    def convert_tokens_to_string(self, toks):
        return "".join(toks)

def make_input_text(core: str, L: int) -> str:
    sc = " ".join(core)
    return (
        f"<CORE_HEAD> {sc} </CORE_HEAD> "
        f"<CORE_TAIL> {sc} </CORE_TAIL> "
        f"<LEN> {L} </LEN>"
    )

def _count_letters_in_ids(tokenizer, ids):
    toks = tokenizer.convert_ids_to_tokens(ids.tolist())
    txt  = tokenizer.convert_tokens_to_string(toks)
    return sum(1 for ch in txt if ch in LETTER_SET)

class BlockEosUntilLetters(torch.nn.Module):
    def __init__(self, tokenizer, eos_id, span_lens):
        super().__init__()
        self.tokenizer = tokenizer
        self.eos_id = eos_id
        self.span_lens = span_lens
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        B = input_ids.size(0)
        for b in range(B):
            Ls  = self.span_lens[b] if isinstance(self.span_lens, list) else int(self.span_lens)
            cnt = _count_letters_in_ids(self.tokenizer, input_ids[b])
            if cnt < Ls:
                scores[b, self.eos_id] = -float("inf")
            elif cnt >= Ls:
                scores[b, :] = -float("inf")
                scores[b, self.eos_id] = 0.0
        return scores

class StopAtLetters(torch.nn.Module):
    def __init__(self, tokenizer, span_lens):
        super().__init__()
        self.tokenizer = tokenizer
        self.span_lens = span_lens
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        B = input_ids.size(0)
        for b in range(B):
            Ls  = self.span_lens[b] if isinstance(self.span_lens, list) else int(self.span_lens)
            cnt = _count_letters_in_ids(self.tokenizer, input_ids[b])
            if cnt < Ls:
                return False
        return True
# ---------------------------------------------------------------------

def load_c2c_model(checkpoint_path: str, device: str = None):
    """
    Load tokenizer + model with the same config as training and load state_dict from .pt.
    """
    tokenizer = CharTokenizer()
    config = T5Config(
        vocab_size=len(tokenizer.vocab),
        d_model=256, d_ff=512, num_layers=4, num_decoder_layers=4, num_heads=4,
        dropout_rate=0.3,
        layer_norm_epsilon=1e-6,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.pad_token_id,
    )
    model = T5ForConditionalGeneration(config)
    model.config.decoder_start_token_id = tokenizer.pad_token_id

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    try:
        model.load_state_dict(ckpt, strict=True)
    except Exception:
        # In case the .pt is a full model or keys are prefixed
        model.load_state_dict(ckpt, strict=False)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, tokenizer, device

def sample_c2c_dual(core: str,
                    n_greedy: int,
                    n_sampled: int,
                    checkpoint_path: str,
                    span_len: int = None,
                    temperature: float = 1.0,
                    top_p: float = 0.95,
                    top_k: int = 0,
                    max_length: int = 128,
                    seed: int = None):
    """
    Generate both greedy and sampled cyclic peptides from the trained C2C model.

    Args:
        core: core amino-acid sequence (e.g., "ACDFG")
        n_greedy: number of greedy outputs (do_sample=False)
        n_sampled: number of stochastic samples (do_sample=True)
        checkpoint_path: path to .pt model checkpoint
        span_len: length of predicted span (default = len(core))
        temperature/top_p/top_k: sampling parameters for do_sample=True
        max_length: total token limit per generation
        seed: random seed for reproducibility

    Returns:
        dict with:
          - 'greedy_spans', 'sampled_spans'
          - 'greedy_assembled', 'sampled_assembled'
    """
    if seed is not None:
        random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    model, tokenizer, device = load_c2c_model(checkpoint_path)

    if span_len is None:
        span_len = len(core)

    def _prepare_input(n_batch):
        prompt = make_input_text(core, span_len)
        encs = [tokenizer.encode(prompt, add_eos=False) for _ in range(n_batch)]
        max_in = max(len(x) for x in encs)
        input_ids = torch.full((n_batch, max_in), tokenizer.pad_token_id, dtype=torch.long)
        attn_mask = torch.zeros((n_batch, max_in), dtype=torch.long)
        for i, ids in enumerate(encs):
            input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            attn_mask[i, :len(ids)] = 1
        return input_ids.to(device), attn_mask.to(device)

    # Common stopping/length control
    eos_id = tokenizer.eos_token_id
    processors = lambda n: LogitsProcessorList([BlockEosUntilLetters(tokenizer, eos_id, [span_len]*n)])
    stoppers   = lambda n: StoppingCriteriaList([StopAtLetters(tokenizer, [span_len]*n)])

    # ---------- Greedy ----------
    if n_greedy > 0:
        ids_g, mask_g = _prepare_input(n_greedy)
        gen_g = model.generate(
            input_ids=ids_g,
            attention_mask=mask_g,
            max_length=max_length,
            do_sample=False,
            logits_processor=processors(n_greedy),
            stopping_criteria=stoppers(n_greedy),
        )
        txt_g = tokenizer.batch_decode(gen_g, skip_special_tokens=True)
        greedy_spans = ["".join(ch for ch in t if ch in LETTER_SET)[:span_len] for t in txt_g]
    else:
        greedy_spans = []

    # ---------- Sampling ----------
    if n_sampled > 0:
        ids_s, mask_s = _prepare_input(n_sampled)
        gen_s = model.generate(
            input_ids=ids_s,
            attention_mask=mask_s,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k if top_k > 0 else None,
            logits_processor=processors(n_sampled),
            stopping_criteria=stoppers(n_sampled),
        )
        txt_s = tokenizer.batch_decode(gen_s, skip_special_tokens=True)
        sampled_spans = ["".join(ch for ch in t if ch in LETTER_SET)[:span_len] for t in txt_s]
    else:
        sampled_spans = []

    # ---------- Assemble ----------
    greedy_assembled  = [core + s for s in greedy_spans]
    sampled_assembled = [core + s for s in sampled_spans]

    return {
        "greedy_spans": greedy_spans,
        "sampled_spans": sampled_spans,
        "greedy_assembled": greedy_assembled,
        "sampled_assembled": sampled_assembled,
    }


out = sample_c2c_dual(
    core=core,
    n_greedy=1,
    n_sampled=num_sample - 1,
    checkpoint_path='./c2c_model.pt',
    span_len=span_len,
    temperature=1.0,
    top_p=0.9,
    seed=42,
)


greedy = out["greedy_spans"]
sample = out["sampled_spans"]

file_out = open('./output/predict.fasta', 'w')

n = 1
for i in greedy:
    file_out.write('>pep'+str(n)+'\n')
    file_out.write(core+i+'\n')
    n += 1

for i in sample:
    file_out.write('>pep'+str(n)+'\n')
    file_out.write(core+i+'\n')
    n += 1

