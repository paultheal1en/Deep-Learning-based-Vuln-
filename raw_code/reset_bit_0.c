static inline void reset_bit(uint32_t *field, int bit)

{

    field[bit >> 5] &= ~(1 << (bit & 0x1F));

}
