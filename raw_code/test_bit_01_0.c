static inline int test_bit(unsigned int bit, const unsigned long *map)

{

    return !!((map)[(bit) / BITS_PER_LONG] & (1UL << ((bit) % BITS_PER_LONG)));

}
