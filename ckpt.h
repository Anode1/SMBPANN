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

/* Warm-start NET (already built and net_init'd) from the checkpoint at PATH: for
 * every layer that index-aligns and is weight-compatible (same kind and same
 * weight/bias buffer sizes), copy the saved weights and biases over the random
 * init; leave every other layer as initialized. So an unchanged layer inherits its
 * parent's training and a mutated or added layer starts fresh. Returns 0, or -1 on
 * a missing or malformed file (NET is left as it was on failure of a later layer).*/
int ckpt_warmstart(Net *net, const char *path);

#endif /* SMB_CKPT_H */
