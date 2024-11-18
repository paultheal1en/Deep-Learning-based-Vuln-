static void write_dt(void *ptr, unsigned long addr, unsigned long limit,

                     int flags)

{

    unsigned int e1, e2;

    uint32_t *p;

    e1 = (addr << 16) | (limit & 0xffff);

    e2 = ((addr >> 16) & 0xff) | (addr & 0xff000000) | (limit & 0x000f0000);

    e2 |= flags;

    p = ptr;

    p[0] = tswapl(e1);

    p[1] = tswapl(e2);

}
