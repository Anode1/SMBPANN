/* ckpt.h -- network checkpoints: save trained weights, warm-start from them.
 *
 * The search normally trains every candidate from scratch, so a genome carries
 * only architecture, not the learned solution. Checkpoints let a child INHERIT its
 * parent's trained weights (Lamarckian inheritance): a mutation changes one small
 * thing, so most layers are unchanged and keep their trained weights, and only the
 * changed part is learned afresh. A warm-started child resumes from a good state
 * instead of restarting, which converges faster. See the paper (weight inheritance).
 *
 * The format is plain text (one value per line), so a checkpoint crosses the shell
 * coordinator's process boundary as an ordinary file, like everything else here.
 */
#ifndef SMB_CKPT_H
#define SMB_CKPT_H

#include "net.h"

/* Write NET's architecture and trained weights to PATH. Returns 0, or -1 on a
 * file error. */
int ckpt_save(const Net *net, const char *path);

/* Warm-start NET (already built and net_init'd) from the checkpoint at PATH.
 * Inheritance is all-or-nothing: only if the checkpoint's topology matches NET
 * *exactly* -- same layer count, and every layer the same kind, dims, filters and
 * kernel -- are the saved weights and biases copied over the random init. Any
 * structural difference (a mutated width, an inserted/pruned/flipped layer, a
 * crossover splice) makes a positional copy meaningless -- weights would land in a
 * layer they were not trained in -- so the whole checkpoint is refused and NET
 * keeps its fresh init (the child trains from scratch). Returns 0 on success or on
 * a clean refusal, or -1 on a missing or malformed file. */
int ckpt_warmstart(Net *net, const char *path);

#endif /* SMB_CKPT_H */
