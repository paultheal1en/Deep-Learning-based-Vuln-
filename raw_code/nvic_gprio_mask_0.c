static inline uint32_t nvic_gprio_mask(NVICState *s)

{

    return ~0U << (s->prigroup + 1);

}
