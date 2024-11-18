static av_always_inline int simple_limit(uint8_t *p, ptrdiff_t stride, int flim)

{

    LOAD_PIXELS

    return 2 * FFABS(p0 - q0) + (FFABS(p1 - q1) >> 1) <= flim;

}
