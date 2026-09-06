"""Knowledge-tracing model zoo.

Every model honors the protocol: `predict_prefixes(skills, corrects)` returns
p_t for t=1..T-1 using history 0..t-1 only. Fitting uses train students only.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression

from ..core.model import p_correct, update as hybrid_update
from ..core.types import ConceptSnapshot, GlobalParams, ItemSpec
from .data import KTDataset


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


# ----------------------------------------------------------------- baselines --

class RandomBaseline:
    name = "random"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, ds: KTDataset):
        return self

    def predict_prefixes(self, skills, corrects):
        # position-indexed draws: prediction t depends only on (seed, t), so
        # prefixes and future perturbations leave it untouched
        rng = np.random.default_rng(self.seed)
        return rng.uniform(0.01, 0.99, size=len(skills) - 1)


class MajorityBaseline:
    name = "majority_global"

    def __init__(self):
        self.p = 0.5

    def fit(self, ds: KTDataset):
        y = np.concatenate([ds.seq(i)[1] for i in ds.subset("train")])
        self.p = float(y.mean())
        return self

    def predict_prefixes(self, skills, corrects):
        return np.full(len(skills) - 1, self.p)


class SkillMajorityBaseline:
    name = "majority_skill"

    def __init__(self):
        self.p = None

    def fit(self, ds: KTDataset):
        S = ds.n_skills
        self.S = S
        hits = np.zeros(S)
        seen = np.zeros(S)
        for i in ds.subset("train"):
            sk, y = ds.seq(i)
            np.add.at(seen, sk, 1)
            np.add.at(hits, sk, y)
        self.p = np.where(seen > 0, hits / np.maximum(seen, 1), 0.5)
        return self

    def predict_prefixes(self, skills, corrects):
        return self.p[skills[1:]]


# ------------------------------------------------------------- PFA features --

def build_pfa_features(skills: np.ndarray, corrects: np.ndarray, n_skills: int):
    """Row t (0..T-1) contains history BEFORE position t: per-skill success and
    failure counts so far plus the skill intercept. Returns sparse COO parts."""
    T = len(skills)
    rows, cols, vals = [], [], []
    succ = np.zeros(n_skills, dtype=np.float64)
    fail = np.zeros(n_skills, dtype=np.float64)
    for t in range(T):
        s = skills[t]
        rows += [t, t, t]
        cols += [s, n_skills + s, 2 * n_skills + s]
        vals += [succ[s], fail[s], 1.0]
        if corrects[t]:
            succ[s] += 1
        else:
            fail[s] += 1
    return rows, cols, vals, T


class PFALogistic:
    """PFA (Pavlik & Anderson 2005) fit with regularized logistic regression:
    logit p = gamma_s + beta_s*succ_s - delta_s*fail_s (skill-level KCs)."""
    name = "pfa_logistic"

    def __init__(self, C: float = 1.0):
        self.C = C
        self.clf = None
        self.S = 0

    def fit(self, ds: KTDataset):
        S = ds.n_skills
        self.S = S
        rows, cols, vals, ys = [], [], [], []
        for i in ds.subset("train"):
            sk, y = ds.seq(i)
            r, c, v, _ = build_pfa_features(sk, y, S)
            base = len(ys)
            rows.extend(rr + base for rr in r)
            cols.extend(c)
            vals.extend(v)
            ys.extend(y.tolist())
        X = sparse.csr_matrix((vals, (rows, cols)), shape=(len(ys), 3 * S))
        self.clf = LogisticRegression(C=self.C, max_iter=2000, solver="lbfgs")
        self.clf.fit(X, np.asarray(ys))
        return self

    def predict_prefixes(self, skills, corrects):
        r, c, v, T = build_pfa_features(skills, corrects, self.S)
        X = sparse.csr_matrix((v, (r, c)), shape=(T, 3 * self.S))
        return self.clf.predict_proba(X)[:, 1][1:]


class LogisticRich:
    """Logistic regression with richer, strictly-history features:
    PFA counts + recency + per-skill accuracy + student accuracy + position."""
    name = "logistic_rich"

    EXTRA = 6

    def __init__(self, C: float = 1.0):
        self.C = C
        self.clf = None
        self.S = 0

    def build(self, skills, corrects, S):
        T = len(skills)
        rows, cols, vals = [], [], []
        succ = np.zeros(S)
        fail = np.zeros(S)
        last_seen = np.full(S, -1, dtype=np.int64)
        tot_c = tot_n = 0
        for t in range(T):
            s = skills[t]
            base = 3 * S
            attempts_s = succ[s] + fail[s]
            acc_s = succ[s] / attempts_s if attempts_s > 0 else 0.5
            rec = math.log1p(t - last_seen[s] - 1) if last_seen[s] >= 0 else 0.0
            acc_stu = tot_c / tot_n if tot_n > 0 else 0.5
            feats = [rec, math.log1p(attempts_s), acc_s, acc_stu,
                     math.log1p(tot_n), math.log1p(t)]
            rows += [t, t, t]
            cols += [s, S + s, 2 * S + s]
            vals += [succ[s], fail[s], 1.0]
            for k, fv in enumerate(feats):
                rows.append(t)
                cols.append(base + k)
                vals.append(fv)
            if corrects[t]:
                succ[s] += 1
                tot_c += 1
            else:
                fail[s] += 1
            tot_n += 1
            last_seen[s] = t
        return rows, cols, vals, T

    def fit(self, ds: KTDataset):
        self.S = S = ds.n_skills
        rows, cols, vals, ys = [], [], [], []
        for i in ds.subset("train"):
            sk, y = ds.seq(i)
            r, c, v, _ = self.build(sk, y, S)
            base = len(ys)
            rows.extend(rr + base for rr in r)
            cols.extend(c)
            vals.extend(v)
            ys.extend(y.tolist())
        X = sparse.csr_matrix((vals, (rows, cols)), shape=(len(ys), 3 * S + self.EXTRA))
        self.clf = LogisticRegression(C=self.C, max_iter=2000, solver="lbfgs")
        self.clf.fit(X, np.asarray(ys))
        return self

    def predict_prefixes(self, skills, corrects):
        r, c, v, T = self.build(skills, corrects, self.S)
        X = sparse.csr_matrix((v, (r, c)), shape=(T, 3 * self.S + self.EXTRA))
        return self.clf.predict_proba(X)[:, 1][1:]


# ------------------------------------------------------------------- BKT ----

class BKTModel:
    """Per-skill Bayesian Knowledge Tracing fit with EM (Corbett & Anderson 1994)."""
    name = "bkt_em"

    def __init__(self, n_iter: int = 25, tol: float = 1e-4):
        self.n_iter, self.tol = n_iter, tol
        self.params: dict[int, tuple] = {}

    def _em(self, seqs: list[np.ndarray]):
        L0, T, G, Ss = 0.2, 0.1, 0.2, 0.1
        for _ in range(self.n_iter):
            num_L0 = den_L0 = num_T = den_T = num_G = den_G = num_S = den_S = 0.0
            for y in seqs:
                n = len(y)
                alpha = np.zeros((n, 2))
                beta = np.zeros((n, 2))
                emit = np.where(y[:, None] == 1,
                                np.array([G, 1 - Ss]), np.array([1 - G, Ss]))
                pi = np.array([1 - L0, L0])
                alpha[0] = pi * emit[0]
                tr = np.array([[1 - T, T], [0.0, 1.0]])
                for t in range(1, n):
                    alpha[t] = (alpha[t - 1] @ tr) * emit[t]
                    alpha[t] /= alpha[t].sum() or 1e-12
                beta[-1] = 1.0
                for t in range(n - 2, -1, -1):
                    beta[t] = tr @ (emit[t + 1] * beta[t + 1])
                    beta[t] /= beta[t].sum() or 1e-12
                gamma = alpha * beta
                gamma /= gamma.sum(axis=1, keepdims=True)
                num_L0 += gamma[0, 1]
                den_L0 += 1
                for t in range(n - 1):
                    xi = np.outer(alpha[t], (tr @ (emit[t + 1] * beta[t + 1])))
                    xi /= xi.sum() or 1e-12
                    num_T += xi[0, 1]
                    den_T += gamma[t, 0]
                m1 = y == 1
                m0 = ~m1
                num_G += np.where(m1, gamma[:, 0], 0).sum()
                den_G += gamma[:, 0].sum()
                num_S += np.where(m0, gamma[:, 1], 0).sum()
                den_S += gamma[:, 1].sum()
            new = (
                np.clip(num_L0 / max(den_L0, 1e-9), 0.01, 0.99),
                np.clip(num_T / max(den_T, 1e-9), 0.01, 0.60),
                np.clip(num_G / max(den_G, 1e-9), 0.03, 0.45),
                np.clip(num_S / max(den_S, 1e-9), 0.03, 0.45),
            )
            if max(abs(a - b) for a, b in zip(new, (L0, T, G, Ss))) < self.tol:
                L0, T, G, Ss = new
                break
            L0, T, G, Ss = new
        return L0, T, G, Ss

    def fit(self, ds: KTDataset):
        by_skill: dict[int, list[np.ndarray]] = {}
        for i in ds.subset("train"):
            sk, y = ds.seq(i)
            order = np.argsort(sk, kind="stable")
            sks, ys = sk[order], y[order]
            bounds = np.searchsorted(sks, np.arange(ds.n_skills + 1))
            for s in range(ds.n_skills):
                if bounds[s + 1] > bounds[s]:
                    by_skill.setdefault(s, []).append(ys[bounds[s]:bounds[s + 1]])
        self.params = {}
        for s, seqs in by_skill.items():
            if sum(len(q) for q in seqs) < 30:
                self.params[s] = (0.2, 0.1, 0.2, 0.1)
                continue
            self.params[s] = self._em(seqs)
        self._default = (0.2, 0.1, 0.2, 0.1)
        return self

    def predict_prefixes(self, skills, corrects):
        T = len(skills)
        out = np.empty(T - 1)
        post: dict[int, float] = {}
        for t in range(T - 1):
            # 1. incorporate the observation at position t (history 0..t)
            s_now, y = int(skills[t]), int(corrects[t])
            L0, Tr, G, Sl = self.params.get(s_now, self._default)
            cur = post.get(s_now, L0)
            if y == 1:
                num, den = cur * (1 - Sl), cur * (1 - Sl) + (1 - cur) * G
            else:
                num, den = cur * Sl, cur * Sl + (1 - cur) * (1 - G)
            post[s_now] = num / den if den > 0 else cur
            # 2. predict the NEXT occurrence of skills[t+1]: transition, then emit
            s_nxt = int(skills[t + 1])
            L0n, Trn, Gn, Sln = self.params.get(s_nxt, self._default)
            pl = post.get(s_nxt, L0n)
            pl_next = pl + (1 - pl) * Trn
            out[t] = pl_next * (1 - Sln) + (1 - pl_next) * Gn
        return out


# ------------------------------------------------------------------- IRT ----

class IRTModel:
    """2PL IRT with skills as items. Static abilities: no learning dynamics.
    Unseen (test-split) students get the population prior theta=0 — an honest
    cold-start handicap, documented as such."""
    name = "irt_2pl"

    def __init__(self, n_steps: int = 300, lr: float = 0.05, l2: float = 1e-3):
        self.n_steps, self.lr, self.l2 = n_steps, lr, l2

    def fit(self, ds: KTDataset):
        idx = ds.subset("train")
        u_map = {u: k for k, u in enumerate(ds.users[idx])}
        N, S = len(idx), ds.n_skills
        us, ss, ys = [], [], []
        for i in idx:
            sk, y = ds.seq(i)
            us.extend([u_map[ds.users[i]]] * len(sk))
            ss.extend(sk.tolist())
            ys.extend(y.tolist())
        us, ss, ys = np.asarray(us), np.asarray(ss), np.asarray(ys, dtype=np.float64)
        theta = np.zeros(N)
        b = np.zeros(S)
        a = np.ones(S)
        for _ in range(self.n_steps):
            z = a[ss] * (theta[us] - b[ss])
            p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
            r = ys - p
            g_th = np.bincount(us, weights=a[ss] * r, minlength=N)
            g_b = np.bincount(ss, weights=-a[ss] * r, minlength=S)
            g_a = np.bincount(ss, weights=(theta[us] - b[ss]) * r, minlength=S)
            theta += self.lr * (g_th - self.l2 * theta)
            b += self.lr * (g_b - self.l2 * b)
            a = np.clip(a + self.lr * (g_a - self.l2 * (a - 1)), 0.25, 3.0)
            theta -= theta.mean()
        self.theta_train = dict(zip(ds.users[idx], theta))
        self.a, self.b = a, b
        return self

    def predict_prefixes(self, skills, corrects, user=None):
        th = self.theta_train.get(user, 0.0) if user is not None else 0.0
        return 1 / (1 + np.exp(-np.clip(self.a[skills[1:]] * (th - self.b[skills[1:]]), -30, 30)))


# ------------------------------------------------------- production hybrid --

class HybridProduction:
    """The deployed PFA/IRT hybrid (alpa.core.model), used sequentially.
    Item difficulties are calibrated once from train correct rates (b_s chosen so
    a theta=0 student reproduces the rate); discrimination fixed at 1."""
    name = "hybrid_production"

    def __init__(self):
        self.gp = GlobalParams()
        self.b = None

    def fit(self, ds: KTDataset):
        hits = np.zeros(ds.n_skills)
        seen = np.zeros(ds.n_skills)
        for i in ds.subset("train"):
            sk, y = ds.seq(i)
            np.add.at(seen, sk, 1)
            np.add.at(hits, sk, y)
        rate = np.where(seen > 0, hits / np.maximum(seen, 1), 0.5)
        self.b = -np.vectorize(_logit)(rate)
        return self

    def predict_prefixes(self, skills, corrects):
        states: dict[int, ConceptSnapshot] = {}
        out = np.empty(len(skills) - 1)
        for t in range(len(skills) - 1):
            s_next = skills[t + 1]
            if s_next not in states:
                states[s_next] = ConceptSnapshot(s_next, f"s{s_next}", theta=0.0, var=1.0)
            item = ItemSpec(item_id=int(s_next), stem="", choices=("a", "b"),
                            correct_index=0, concept_ids=(int(s_next),),
                            b=float(self.b[s_next]), a=1.0)
            out[t] = p_correct(states[s_next], item, 0.0, self.gp)
            s_now = skills[t]
            if s_now not in states:
                states[s_now] = ConceptSnapshot(s_now, f"s{s_now}", theta=0.0, var=1.0)
            item_now = ItemSpec(item_id=int(s_now), stem="", choices=("a", "b"),
                                correct_index=0, concept_ids=(int(s_now),),
                                b=float(self.b[s_now]), a=1.0)
            states[s_now] = hybrid_update(states[s_now], item_now, int(corrects[t]),
                                          0.0, self.gp)
        return out


# ------------------------------------------------------------- torch models --

def _torch():
    import torch
    return torch


def device():
    torch = _torch()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _TorchKT:
    """Shared training/inference for sequence models (GRU-KT and attention-KT)."""

    def __init__(self, d_model: int = 64, chunk: int = 200, batch: int = 32,
                 epochs: int = 12, lr: float = 1e-3, patience: int = 2,
                 seed: int = 42, checkpoint_dir: str | None = None):
        self.d_model, self.chunk, self.batch = d_model, chunk, batch
        self.epochs, self.lr, self.patience, self.seed = epochs, lr, patience, seed
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.net, self.S = None, 0
        self.history: list[dict] = []

    def _make_net(self, S):
        raise NotImplementedError

    def _extra_features(self, skills_chunk, corrects_prefix, skills_prefix):
        """Optional per-step auxiliary features (T, d_extra), history-only."""
        return None

    def _encode_input(self, net, skills, corrects):
        torch = _torch()
        x = skills + self.S * corrects.long()
        return net.embed(x)

    def _forward_logits(self, net, skills, corrects):
        raise NotImplementedError

    def fit(self, ds: KTDataset):
        torch = _torch()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.S = ds.n_skills
        self.net = self._make_net(ds.n_skills).to(device())
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        lossf = torch.nn.BCEWithLogitsLoss(reduction="none")
        pad = 2 * self.S

        def chunks_of(split):
            out = []
            for i in ds.subset(split):
                sk, y = ds.seq(i)
                for lo in range(0, len(sk), self.chunk):
                    c_sk = sk[lo:lo + self.chunk]
                    if len(c_sk) < 2:
                        continue
                    out.append((torch.from_numpy(c_sk.astype(np.int64)),
                                torch.from_numpy(y[lo:lo + self.chunk].astype(np.float32)),
                                self._extra_features(c_sk, y[:lo + len(c_sk)],
                                                     sk[:lo + len(c_sk)])))
            return out

        train_ch = chunks_of("train")

        def eval_logloss(split):
            self.net.eval()
            total = nll = 0.0
            with torch.no_grad():
                for i in ds.subset(split):
                    sk, y = ds.seq(i)
                    if len(sk) < 2:
                        continue
                    skt = torch.from_numpy(sk.astype(np.int64)).to(device())
                    yt = torch.from_numpy(y.astype(np.float32)).to(device())
                    ft = self._extra_features(sk, y, sk)
                    ft_t = torch.from_numpy(ft).to(device()).unsqueeze(0) if ft is not None else None
                    logits = self._forward_logits(self.net, skt.unsqueeze(0), yt.unsqueeze(0), ft_t)[0]
                    p = torch.sigmoid(logits[: len(sk) - 1]).gather(
                        1, skt[1:].unsqueeze(1)).squeeze(1)
                    nll += float(-torch.sum(yt[1:] * torch.log(p.clamp(min=1e-7))
                                            + (1 - yt[1:]) * torch.log((1 - p).clamp(min=1e-7))))
                    total += len(sk) - 1
            self.net.train()
            return nll / max(total, 1)

        rng = np.random.default_rng(self.seed)
        best_val, best_state, bad = float("inf"), None, 0
        for epoch in range(self.epochs):
            order = rng.permutation(len(train_ch))
            running = 0.0
            for b0 in range(0, len(order), self.batch):
                sel = order[b0:b0 + self.batch]
                batch = [train_ch[k] for k in sel]
                L = max(len(s) for s, _, _ in batch)
                has_f = batch[0][2] is not None
                sk_b = torch.full((len(batch), L), pad, dtype=torch.int64).to(device())
                y_b = torch.zeros(len(batch), L).to(device())
                mask = torch.zeros(len(batch), L - 1).to(device())
                f_b = None
                if has_f:
                    d_extra = batch[0][2].shape[1]
                    f_b = torch.zeros(len(batch), L, d_extra).to(device())
                for r, (s, y, f) in enumerate(batch):
                    sk_b[r, :len(s)] = s
                    y_b[r, :len(y)] = y
                    mask[r, :len(s) - 1] = 1.0
                    if has_f:
                        f_b[r, :len(s)] = torch.from_numpy(f).to(device())
                opt.zero_grad()
                logits = self._forward_logits(self.net, sk_b, y_b, f_b)   # (B, L-1, S)
                tgt = sk_b[:, 1:].clamp(max=self.S - 1)             # pad rows masked below
                picked = logits.gather(2, tgt.unsqueeze(2)).squeeze(2)
                loss = (lossf(picked, y_b[:, 1:]) * mask).sum() / mask.sum()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                opt.step()
                running += float(loss.detach()) * int(mask.sum())
            val = eval_logloss("val")
            self.history.append({"epoch": epoch, "train_logloss": running / max(1, sum(len(s) - 1 for s, _, _ in train_ch)),
                                 "val_logloss": val})
            if val < best_val - 1e-4:
                best_val, bad = val, 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.best_val_logloss = best_val
        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            path = self.checkpoint_dir / f"{self.name}.pt"
            torch.save({"name": self.name, "state": self.net.state_dict(),
                        "S": self.S, "d_model": self.d_model, "seed": self.seed,
                        "best_val_logloss": best_val,
                        "history": self.history,
                        "device": str(device())}, path)
            self.checkpoint_path = str(path)
        return self

    def load(self, ds: KTDataset, path: str):
        torch = _torch()
        ck = torch.load(path, map_location=device(), weights_only=True)
        self.S, self.d_model = ck["S"], ck["d_model"]
        self.net = self._make_net(self.S).to(device())
        self.net.load_state_dict(ck["state"])
        self.net.eval()
        return self

    def predict_prefixes(self, skills, corrects):
        torch = _torch()
        self.net.eval()
        with torch.no_grad():
            sk = torch.from_numpy(np.asarray(skills, dtype=np.int64)).to(device())
            y = torch.from_numpy(np.asarray(corrects, dtype=np.int64)).to(device())
            ft = self._extra_features(np.asarray(skills), np.asarray(corrects),
                                      np.asarray(skills))
            ft_t = torch.from_numpy(ft).to(device()).unsqueeze(0) if ft is not None else None
            logits = self._forward_logits(self.net, sk.unsqueeze(0), y.unsqueeze(0), ft_t)[0]
            p = torch.sigmoid(logits[: len(skills) - 1]).gather(
                1, sk[1:].unsqueeze(1)).squeeze(1)
        return p.cpu().numpy()


class GRUKT(_TorchKT):
    """DKT-style GRU knowledge tracer (Piech et al. 2015 formulation)."""
    name = "gru_kt"

    def _make_net(self, S):
        torch = _torch()

        class Net(torch.nn.Module):
            def __init__(self, S, d):
                super().__init__()
                self.embed = torch.nn.Embedding(2 * S + 1, d, padding_idx=2 * S)
                self.gru = torch.nn.GRU(d, d, batch_first=True)
                self.head = torch.nn.Linear(d, S)

            def forward(self, x_in):
                h, _ = self.gru(x_in)
                return self.head(h)

        return Net(S, self.d_model)

    def _forward_logits(self, net, sk_b, y_b, f_b=None):
        x = sk_b[:, :-1] + self.S * y_b[:, :-1].long()
        return net(net.embed(x))


class ConceptAwareGRU(_TorchKT):
    """GRU-KT conditioned on a GIVEN prerequisite graph: each step's input is
    augmented with the running success rate of the current skill's prerequisites
    and the skill's own running rate (both strictly history-only). The test
    question: does graph structure improve sequence prediction where it exists?
    Evaluated on the synthetic dataset, which has planted prerequisite coupling;
    ASSISTments carries no graph, so running it there would be meaningless."""
    name = "concept_aware_gru"
    D_EXTRA = 2

    def __init__(self, prereqs: dict[int, list[int]], n_skills: int, **kw):
        super().__init__(**kw)
        self.prereqs = prereqs
        self._n_skills = n_skills

    def _extra_features(self, skills_chunk, corrects_prefix, skills_prefix):
        n = len(skills_prefix)
        T = len(skills_chunk)
        own = np.full(n, 0.5)
        pre = np.full(n, 0.5)
        succ = np.zeros(self._n_skills)
        att = np.zeros(self._n_skills)
        for t in range(n):
            s = int(skills_prefix[t])
            if att[s] > 0:
                own[t] = succ[s] / att[s]
            ps = self.prereqs.get(s, [])
            if ps:
                pre[t] = float(np.mean([succ[p] / att[p] if att[p] > 0 else 0.5
                                        for p in ps]))
            if corrects_prefix[t]:
                succ[s] += 1
            att[s] += 1
        return np.stack([own, pre], axis=1)[n - T:].astype(np.float32)

    def _make_net(self, S):
        torch = _torch()

        class Net(torch.nn.Module):
            def __init__(self, S, d, d_extra):
                super().__init__()
                self.embed = torch.nn.Embedding(2 * S + 1, d, padding_idx=2 * S)
                self.fproj = torch.nn.Linear(d_extra, d)
                self.gru = torch.nn.GRU(d, d, batch_first=True)
                self.head = torch.nn.Linear(d, S)

            def forward(self, x_emb):
                h, _ = self.gru(x_emb)
                return self.head(h)

        return Net(S, self.d_model, self.D_EXTRA)

    def _forward_logits(self, net, sk_b, y_b, f_b=None):
        x = sk_b[:, :-1] + self.S * y_b[:, :-1].long()
        h_in = net.embed(x)
        if f_b is not None:
            h_in = h_in + net.fproj(f_b[:, :-1])
        return net(h_in)
class DAKT(_TorchKT):
    """Attention-based KT: causal transformer encoder over the interaction
    history (simplified DAKT/AKT family; no item-difficulty term)."""
    name = "dakt_attention"

    def _make_net(self, S):
        torch = _torch()

        class Net(torch.nn.Module):
            def __init__(self, S, d):
                super().__init__()
                self.embed = torch.nn.Embedding(2 * S + 1, d, padding_idx=2 * S)
                self.pos = torch.nn.Embedding(1024, d)
                layer = torch.nn.TransformerEncoderLayer(
                    d, nhead=2, dim_feedforward=2 * d, batch_first=True, dropout=0.1)
                self.enc = torch.nn.TransformerEncoder(layer, num_layers=2)
                self.head = torch.nn.Linear(d, S)

            def forward(self, x_ids, pos):
                h = self.embed(x_ids) + self.pos(pos)
                L = x_ids.size(1)
                mask = torch.triu(torch.ones(L, L, dtype=torch.bool,
                                             device=x_ids.device), diagonal=1)
                h = self.enc(h, mask=mask)
                return self.head(h)

        return Net(S, self.d_model)

    def _forward_logits(self, net, sk_b, y_b, f_b=None):
        torch = _torch()
        x = sk_b[:, :-1] + self.S * y_b[:, :-1].long()
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
        return net(x, pos)
