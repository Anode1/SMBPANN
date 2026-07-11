/* arena.c -- Mark/Release bump allocator. See arena.h. */
#include <stdlib.h>

#include "arena.h"

int arena_init(Arena *a, size_t cap)
{
    if (a == NULL || cap == 0)
        return -1;
    a->base = malloc(cap);
    if (a->base == NULL)
        return -1;
    a->cap = cap;
    a->top = 0;
    return 0;
}

void *arena_alloc(Arena *a, size_t n, size_t align)
{
    size_t start;

    if (align == 0)
        align = 1;
    /* round the current top up to ALIGN, then check the request fits before
     * handing out the address and bumping the top */
    start = ((a->top + align - 1) / align) * align;
    if (start + n > a->cap)
        return NULL;
    a->top = start + n;
    return a->base + start;
}

size_t arena_mark(const Arena *a)         { return a->top; }
void   arena_release(Arena *a, size_t m)  { a->top = m; }
void   arena_reset(Arena *a)              { a->top = 0; }
size_t arena_used(const Arena *a)         { return a->top; }
size_t arena_capacity(const Arena *a)     { return a->cap; }

void arena_free(Arena *a)
{
    if (a == NULL)
        return;
    free(a->base);
    a->base = NULL;
    a->cap = 0;
    a->top = 0;
}
