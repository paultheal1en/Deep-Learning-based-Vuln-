static inline uint8_t *get_seg_base(uint32_t e1, uint32_t e2)

{

    return (uint8_t *)((e1 >> 16) | ((e2 & 0xff) << 16) | (e2 & 0xff000000));

}
