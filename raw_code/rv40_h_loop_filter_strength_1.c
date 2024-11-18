static int rv40_h_loop_filter_strength(uint8_t *src, int stride,

                                       int beta, int beta2, int edge,

                                       int *p1, int *q1)

{

    return rv40_loop_filter_strength(src, stride, 1, beta, beta2, edge, p1, q1);

}
