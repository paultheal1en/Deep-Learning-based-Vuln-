void helper_lsl(void)

{

    unsigned int selector, limit;

    uint32_t e1, e2;



    CC_SRC = cc_table[CC_OP].compute_all() & ~CC_Z;

    selector = T0 & 0xffff;

    if (load_segment(&e1, &e2, selector) != 0)

        return;

    limit = (e1 & 0xffff) | (e2 & 0x000f0000);

    if (e2 & (1 << 23))

        limit = (limit << 12) | 0xfff;

    T1 = limit;

    CC_SRC |= CC_Z;

}
