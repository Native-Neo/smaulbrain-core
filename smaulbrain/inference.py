"""Inference engine using Torch tensors and persistent recurrent state.

Speculative decoding (opt-in via ``speculative=K``) uses the model's MTP
modules to draft tokens after the normally sampled one, verifies them with a
single batched main-model forward, and accepts the matching prefix:

* ``temperature == 0``: greedy output is token-identical to plain decoding
  (the verifier's per-position logits use the exact single-token math).
* ``temperature > 0``: drafts are accepted with the standard min(1, p/q)
  rejection rule and rejects are resampled from the normalized residual
  (p-q)+, so the emitted distribution still equals the target distribution.

MTP draft states are advanced only over verified tokens, keeping the draft
distribution consistent with how the modules were trained. Module 1 uses the
verified pipeline directly; deeper modules (depth >= 3) whose verified state
still lags the frontier are advanced tentatively *inside* the draft pass over
the drafts produced by that same pass, and the tentative states/outputs are
discarded at the end of the iteration, so rejected drafts never pollute the
verified pipeline. ``draft_depth`` (<= ``speculative`` and <= model
``mtp_depth``) reports how many drafts are used per round.
"""
import torch


class InferenceEngine:
    _WINDOW = 256   # max positions processed per batched MTP catch-up pass
    _RING = 2048    # per-depth ring buffer size (hidden/state history)

    def __init__(self,model,seed=0,temperature=1.0,top_k=0,top_p=1.0,speculative=0):
        self.model=model;self.temperature=float(temperature);self.top_k=int(top_k);self.top_p=float(top_p)
        if not 0<self.top_p<=1:raise ValueError("--top-p must be in (0, 1]")
        if self.top_k<0:raise ValueError("--top-k must be >= 0")
        if not self.temperature>=0:raise ValueError("--temperature must be >= 0")
        self.mtp_depth=int(getattr(model,"mtp_depth",0) or 0)
        self.speculative=int(speculative)
        if self.speculative<0:raise ValueError("speculative must be >= 0")
        if self.speculative>self.mtp_depth:
            raise ValueError(f"speculative={self.speculative} exceeds model mtp_depth={self.mtp_depth}")
        self.draft_depth=min(self.speculative,self.mtp_depth)
        self.generator=torch.Generator(device=getattr(model,"device","cpu"));self.generator.manual_seed(int(seed));self.states=model.init_states()
        self._reset_mtp()

    def reset(self):
        self.states=self.model.init_states(); self._reset_mtp()

    # -- persistent state (main recurrent layers + optional MTP pipeline) --
    def export_states(self):
        blob=self.model.serialize_states(self.states)
        if self._mtp_ready:
            base=max(0,(min(self._mtp_pos) if self._mtp_pos else 0)-1)
            lo=max(0,base-self._tok_base)
            blob["mtp"]={"base":int(base),
                         "tokens":list(self._toks[lo:]),
                         "h0":[[int(p),v.tolist()] for p,v in self._h0 if p>=base],
                         "h":[[[int(p),v.tolist()] for p,v in ring if p>=base] for ring in self._h],
                         "states":[None if s is None else s.tolist() for s in self._mtp_states],
                         "pos":list(self._mtp_pos),
                         "unconsumed":bool(self._unconsumed)}
        return blob

    def import_states(self,blob):
        main={k:v for k,v in blob.items() if k!="mtp"}
        self.states=[s.to(self.model.device) for s in self.model.restore_states(main)]
        self._reset_mtp()
        mtp=blob.get("mtp")
        if mtp is None or not self._mtp_ready:
            # Blobs from plain-decoding engines carry no draft pipeline; stay
            # on the plain path (same output distribution) until reset/prefill.
            self._mtp_ready=False
            return
        self._tok_base=int(mtp["base"]); self._toks=[int(t) for t in mtp["tokens"]]
        self._h0=[(int(p),torch.as_tensor(v,dtype=torch.float32)) for p,v in mtp["h0"]]
        self._h=[[(int(p),torch.as_tensor(v,dtype=torch.float32)) for p,v in ring] for ring in mtp["h"]]
        self._mtp_states=[None if s is None else torch.as_tensor(s,dtype=torch.float32) for s in mtp["states"]]
        self._mtp_pos=[int(p) for p in mtp["pos"]]
        # Blob and engine may use different draft depths: missing depths start
        # fresh but positioned at the blob's deepest verified frontier (their
        # inputs are only recorded from `base` on), and depths the importer
        # does not use are dropped so the pipeline lists keep draft_depth rows.
        pad=self._mtp_pos[-1] if self._mtp_pos else 0
        k=self.draft_depth
        self._mtp_states=(self._mtp_states+[None]*k)[:k]
        self._h=(self._h+[[] for _ in range(k)])[:k]
        self._mtp_pos=(self._mtp_pos+[pad]*k)[:k]
        # `unconsumed` records whether the last recorded token still awaits its
        # model step; blobs without it come from engines that always resumed as
        # if it were true.
        self._unconsumed=bool(mtp.get("unconsumed",True))

    def _reset_mtp(self):
        k=self.draft_depth
        self._mtp_ready=bool(k>0)
        self._mtp_K=k
        self._mtp_states=[None]*k; self._mtp_pos=[0]*k
        self._tok_base=0; self._toks=[]
        self._h0=[]; self._h=[[ ] for _ in range(k)]
        self._unconsumed=False

    def _ring_put(self,ring,pos,vec):
        pos=int(pos)
        if ring and ring[-1][0]==pos: ring[-1]=(pos,vec.detach())
        else: ring.append((pos,vec.detach()))
        while len(ring)>self._RING: ring.pop(0)

    @staticmethod
    def _ring_get(ring,pos):
        for p,v in ring:
            if p==pos:return v
        return None

    def _tok_at(self,pos):
        i=pos-self._tok_base
        return self._toks[i] if 0<=i<len(self._toks) else None

    def _trim(self):
        if not self._mtp_pos:return
        keep=max(0,min(self._mtp_pos)-1)
        if keep>self._tok_base:
            drop=min(keep-self._tok_base,len(self._toks))
            if drop>0:
                del self._toks[:drop]; self._tok_base+=drop
        for ring in [self._h0]+self._h:
            while ring and ring[0][0]<keep: ring.pop(0)

    def _mtp_catch_up(self,upto=None):
        """Advance MTP states over consecutive verified tokens, up to `upto`
        (absolute position). Draft passes never mutate these states."""
        if not self._mtp_ready:return
        if upto is None: upto=self._tok_base+len(self._toks)
        while True:
            moved=False
            for k in range(1,self._mtp_K+1):
                src=self._h0 if k==1 else self._h[k-2]
                pos=self._mtp_pos[k-1]; end=min(upto,pos+self._WINDOW)
                span=[]; j=pos
                while j<end:
                    if self._tok_at(j+k) is None or self._ring_get(src,j) is None: break
                    span.append(j); j+=1
                if not span: continue
                H=torch.stack([self._ring_get(src,x) for x in span])
                tids=torch.tensor([self._tok_at(x+k) for x in span],dtype=torch.long)
                _logits,Y,s=self.model._mtp_module(k,H,tids,state=self._mtp_states[k-1])
                self._mtp_states[k-1]=s; self._mtp_pos[k-1]=span[-1]+1
                for x,row in zip(span,Y): self._ring_put(self._h[k-1],x,row)
                moved=True
            if not moved: break
        self._trim()

    def prefill(self,tokens,chunk=0):
        toks=list(tokens)
        if not toks:
            # Nothing to consume: an empty prefill must not clear the pending-token
            # flag, or the next resume appends a duplicate token record at the
            # frontier and the MTP catch-up stops forever.
            return self.states
        if not self._mtp_ready and self.draft_depth>0:
            # A blob without a draft pipeline left the engine on the plain path;
            # a fresh prefill starts a new session, so re-enable drafting (as the
            # import_states comment promises).
            self._reset_mtp()
        if not self._mtp_ready:
            if chunk and chunk>0:
                for i in range(0,len(toks),chunk):self.model.forward(toks[i:i+chunk],states=self.states)
            elif toks:self.model.forward(toks,states=self.states)
            return self.states
        # Chunked prefill bounds the hidden-state history retained for MTP
        # drafting. Chunk size already changes MoE capacity semantics, so this
        # does not introduce a new source of chunk dependence.
        step=chunk if chunk and chunk>0 else 512
        for i in range(0,len(toks),step):
            seg=toks[i:i+step]
            base=self._tok_base+len(self._toks)
            _logits,hidden=self.model.hidden_forward(seg,self.states)
            for j in range(hidden.shape[0]): self._ring_put(self._h0,base+j,hidden[j])
            self._toks.extend(int(x) for x in seg)
            self._mtp_catch_up()
        self._unconsumed=False
        return self.states

    # -- sampling (kept identical to plain decoding) --
    def _probs(self,logits):
        if self.temperature==0:
            p=torch.zeros_like(logits); p[int(torch.argmax(logits).item())]=1.0; return p
        l=logits.float()/self.temperature
        if self.top_k>0:
            k=min(self.top_k,l.numel()); threshold=torch.topk(l,k).values[-1];l=l.masked_fill(l<threshold,-torch.inf)
        if self.top_p<1:
            sorted_l,idx=torch.sort(l,descending=True); probs=torch.softmax(sorted_l,dim=0); cumulative=probs.cumsum(0)
            remove=cumulative-probs>=self.top_p; sorted_l=sorted_l.masked_fill(remove,-torch.inf)
            l=torch.full_like(l,-torch.inf).scatter(0,idx,sorted_l)
        return torch.softmax(l,dim=0)

    def _draw(self,probs):
        if self.temperature==0:return int(torch.argmax(probs).item())
        return int(torch.multinomial(probs,1,generator=self.generator).item())

    def _sample(self,logits):
        return self._draw(self._probs(logits))

    def _accept(self,p,q,draft):
        """Speculative acceptance: greedy equality, or min(1, p/q) with a
        normalized (p-q)+ residual sample on rejection. Returns (ok, repl)."""
        if self.temperature==0:
            best=int(torch.argmax(p).item())
            return best==int(draft), (None if best==int(draft) else best)
        pd=float(p[draft].item()); qd=float(q[draft].item())
        if qd<=0:return False,self._draw(p)
        ratio=pd/qd
        if ratio>=1.0:return True,None
        if float(torch.rand((),generator=self.generator).item())<ratio:return True,None
        resid=torch.clamp(p-q,min=0.0); total=float(resid.sum().item())
        if total<=0:return False,self._draw(p)
        return False,self._draw(resid/total)

    def generate(self,prompt_tokens,max_new_tokens=64,stop_ids=()):
        return list(self.generate_stream(prompt_tokens,max_new_tokens,stop_ids))

    def generate_stream(self,prompt_tokens,max_new_tokens=64,stop_ids=()):
        prompt=list(prompt_tokens);self.prefill(prompt);stop=set(stop_ids)
        if not (self._mtp_ready and self.speculative>0):
            last=prompt[-1] if prompt else 0
            for _ in range(max_new_tokens):
                nxt=self._sample(self.model.step_token(last,self.states));last=nxt;yield nxt
                if nxt in stop:break
            return
        # Plain decoding re-consumes the last prefill token (existing
        # convention). When generation resumes after an earlier generate() or
        # import_states(), the recorded last token is still unconsumed.
        last=self._toks[-1] if self._toks else 0
        if not self._unconsumed:
            self._toks.append(int(last))
        self._unconsumed=True
        yield from self._generate_speculative(last,max_new_tokens,stop)

    def _mtp_draft(self,pos):
        """Draft up to `_mtp_K` tokens after the frontier at `pos`.

        Each depth runs one batched step over its tentative span
        [verified_pos_k, pos]: module 1 runs on verified state only, and deeper
        modules whose verified state still lags `pos` are extended over the
        drafts produced by shallower modules in this same pass (e.g. depth 3
        needs tok[p+2] = draft 1 to reach position p), which is why spans are
        processed in depth order. The last row of each step samples the draft
        from that depth's logits. Tentative states and outputs live only for
        this call: rejected drafts are never persisted, and the verified
        pipeline is re-derived each round by `_mtp_catch_up`.
        """
        drafts=[];qs=[]
        K=self._mtp_K
        if K<=0: return drafts,qs
        pending={}                       # absolute position -> tentative token
        out=[dict() for _ in range(K)]   # out[k-1][pos] = tentative h_k output
        def tok_tent(p):
            if p in pending: return pending[p]
            return self._tok_at(p)
        def hprev_tent(k,j):
            if k==1: return self._ring_get(self._h0,j)
            return out[k-2].get(j, self._ring_get(self._h[k-2],j))
        for k in range(1,K+1):
            rows=list(range(self._mtp_pos[k-1],pos+1))
            H=[];T=[];ready=True
            for j in rows:
                hp=hprev_tent(k,j); tv=tok_tent(j+k)
                if hp is None or tv is None: ready=False; break
                H.append(hp);T.append(tv)
            if not ready: break
            logits,Y,_s=self.model._mtp_module(k,torch.stack(H),
                                               torch.tensor(T,dtype=torch.long),
                                               state=self._mtp_states[k-1])
            for j,row in zip(rows,Y): out[k-1][j]=row
            probs=self._probs(logits[-1]); d=self._draw(probs)
            drafts.append(d);qs.append(probs)
            pending[pos+k+1]=d
        return drafts,qs

    def _rollback(self,hist,idx):
        self.states=[hist[l][idx].detach().clone() for l in range(len(self.states))]

    def _generate_speculative(self,last,max_new_tokens,stop):
        remaining=int(max_new_tokens)
        while remaining>0:
            logits,h0=self.model.step_details(last,self.states)
            self._unconsumed=False
            pos=self._tok_base+len(self._toks)-1          # position of `last`
            self._ring_put(self._h0,pos,h0)
            t1=self._sample(logits)
            self._toks.append(int(t1)); self._unconsumed=True
            self._mtp_catch_up(pos)
            drafts,qs=self._mtp_draft(pos)
            remaining-=1; yield t1
            if t1 in stop or remaining==0: return
            if not drafts: last=t1; continue
            K=len(drafts); seq=[t1]+drafts
            vlogits,hidden,hist=self.model.verify_tokens(seq,self.states)
            accepted=0; repl=None
            for j,d in enumerate(drafts):
                ok,r=self._accept(self._probs(vlogits[j]),qs[j],d)
                if ok: accepted+=1
                else: repl=r; break
            if accepted==K: repl=self._draw(self._probs(vlogits[K]))
            # State after t1 + the accepted draft prefix.
            self._rollback(hist,accepted)
            for j in range(accepted):
                self._toks.append(int(drafts[j]))
            # hidden[0] is t1's position (always a true token) and hidden[j]
            # for j <= accepted was computed over accepted tokens only.
            for j in range(accepted+1):
                self._ring_put(self._h0,pos+1+j,hidden[j])
            self._toks.append(int(repl))
            for m,d in enumerate(drafts[:accepted]):
                remaining-=1; yield d
                if d in stop or remaining==0:
                    # Keep the invariant "state covers all but the last
                    # emitted token": roll back and drop unemitted records.
                    # `pos` is absolute, the token list may be trimmed.
                    self._rollback(hist,m)
                    del self._toks[max(0,pos+3+m-self._tok_base):]
                    return
            remaining-=1; yield repl
            if repl in stop or remaining==0: return
            last=repl
