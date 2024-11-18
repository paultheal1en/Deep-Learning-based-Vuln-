static inline int asym_quant(int c, int e, int qbits)

{

    int m;



    c = (((c << e) >> (24 - qbits)) + 1) >> 1;

    m = (1 << (qbits-1));

    if (c >= m)

        c = m - 1;

    av_assert2(c >= -m);

    return c;

}
