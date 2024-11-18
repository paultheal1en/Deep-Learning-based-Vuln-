void do_brinc (void)

{

    uint32_t a, b, d, mask;



    mask = (uint32_t)(-1UL) >> MASKBITS;

    b = T1_64 & mask;

    a = T0_64 & mask;

    d = word_reverse(1 + word_reverse(a | ~mask));

    T0_64 = (T0_64 & ~mask) | (d & mask);

}
