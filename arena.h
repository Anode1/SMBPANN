/* arena.h -- a marker/region (Mark-Release) allocator.
 *
 * A bump allocator over one fixed block: arena_alloc advances a high-water mark,
 * there is no individual free, and arena_release rolls the mark back to a saved
 * point, reclaiming everything allocated since it in O(1). This is the "manual
 * heap with markers" discipline (Turbo Pascal Mark/Release). For the genetic
 * search it is the substrate a generation of candidate networks carves from:
 * save a mark, build the generation, release -- the whole generation freed at
 * once, no per-network teardown, no fragmentation, one bounded allocation for
 * the run.
 *
 * Ported line-for-line from the Ada smbpann-arena package.
 */
#ifndef SMB_ARENA_H
#define SMB_ARENA_H

#include <stddef.h>

typedef struct {
    unsigned char *base;   /* the block, malloc'd once by arena_init */
    size_t         cap;    /* its size in bytes                      */
    size_t         top;    /* bytes handed out so far (the mark)     */
} Arena;

/* Allocate the arena's block (CAP bytes). Returns 0, or -1 on bad args / OOM. */
int arena_init(Arena *a, size_t cap);

/* Carve N bytes, aligned up to ALIGN. Returns a pointer into the block, or NULL
 * if the request does not fit. */
void *arena_alloc(Arena *a, size_t n, size_t align);

/* Save the current high-water mark, to arena_release back to later. */
size_t arena_mark(const Arena *a);

/* Reclaim everything allocated since MARK. O(1); any pointer into the released
 * region is dangling afterwards (the one rule the caller must keep). */
void arena_release(Arena *a, size_t mark);

/* Reclaim everything. */
void arena_reset(Arena *a);

size_t arena_used(const Arena *a);
size_t arena_capacity(const Arena *a);

/* Release the block. NULL-safe; safe on a zeroed or failed arena. */
void arena_free(Arena *a);

#endif /* SMB_ARENA_H */
