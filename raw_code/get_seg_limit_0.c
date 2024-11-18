static inline unsigned int get_seg_limit(uint32_t e1, uint32_t e2)

{

    unsigned int limit;

    limit = (e1 & 0xffff) | (e2 & 0x000f0000);

    if (e2 & DESC_G_MASK)

        limit = (limit << 12) | 0xfff;

    return limit;

}
