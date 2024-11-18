static inline int load_segment(uint32_t *e1_ptr, uint32_t *e2_ptr,

                               int selector)

{

    SegmentCache *dt;

    int index;

    uint8_t *ptr;



    if (selector & 0x4)

        dt = &env->ldt;

    else

        dt = &env->gdt;

    index = selector & ~7;

    if ((index + 7) > dt->limit)

        return -1;

    ptr = dt->base + index;

    *e1_ptr = ldl_kernel(ptr);

    *e2_ptr = ldl_kernel(ptr + 4);

    return 0;

}
