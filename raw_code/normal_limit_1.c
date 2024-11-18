static av_always_inline int normal_limit(uint8_t *p, ptrdiff_t stride,

                                         int E, int I)

{

    LOAD_PIXELS

    return simple_limit(p, stride, E) &&

           FFABS(p3 - p2) <= I &&

           FFABS(p2 - p1) <= I &&

           FFABS(p1 - p0) <= I &&

           FFABS(q3 - q2) <= I &&

           FFABS(q2 - q1) <= I &&

           FFABS(q1 - q0) <= I;

}
