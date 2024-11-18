static inline int16_t logadd(int16_t a, int16_t b)

{

    int16_t c = a - b;

    uint8_t address = FFMIN((ABS(c) >> 1), 255);



    return ((c >= 0) ? (a + latab[address]) : (b + latab[address]));

}
