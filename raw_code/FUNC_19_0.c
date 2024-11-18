static inline void FUNC(idctRowCondDC_extrashift)(int16_t *row, int extra_shift)

#else

static inline void FUNC(idctRowCondDC)(int16_t *row, int extra_shift)

#endif

{

    int a0, a1, a2, a3, b0, b1, b2, b3;



#if HAVE_FAST_64BIT

#define ROW0_MASK (0xffffLL << 48 * HAVE_BIGENDIAN)

    if (((((uint64_t *)row)[0] & ~ROW0_MASK) | ((uint64_t *)row)[1]) == 0) {

        uint64_t temp;

        if (DC_SHIFT - extra_shift >= 0) {

            temp = (row[0] * (1 << (DC_SHIFT - extra_shift))) & 0xffff;

        } else {

            temp = ((row[0] + (1<<(extra_shift - DC_SHIFT-1))) >> (extra_shift - DC_SHIFT)) & 0xffff;

        }

        temp += temp * (1 << 16);

        temp += temp * ((uint64_t) 1 << 32);

        ((uint64_t *)row)[0] = temp;

        ((uint64_t *)row)[1] = temp;

        return;

    }

#else

    if (!(((uint32_t*)row)[1] |

          ((uint32_t*)row)[2] |

          ((uint32_t*)row)[3] |

          row[1])) {

        uint32_t temp;

        if (DC_SHIFT - extra_shift >= 0) {

            temp = (row[0] * (1 << (DC_SHIFT - extra_shift))) & 0xffff;

        } else {

            temp = ((row[0] + (1<<(extra_shift - DC_SHIFT-1))) >> (extra_shift - DC_SHIFT)) & 0xffff;

        }

        temp += temp * (1 << 16);

        ((uint32_t*)row)[0]=((uint32_t*)row)[1] =

            ((uint32_t*)row)[2]=((uint32_t*)row)[3] = temp;

        return;

    }

#endif



    a0 = (W4 * row[0]) + (1 << (ROW_SHIFT + extra_shift - 1));

    a1 = a0;

    a2 = a0;

    a3 = a0;



    a0 += W2 * row[2];

    a1 += W6 * row[2];

    a2 -= W6 * row[2];

    a3 -= W2 * row[2];



    b0 = MUL(W1, row[1]);

    MAC(b0, W3, row[3]);

    b1 = MUL(W3, row[1]);

    MAC(b1, -W7, row[3]);

    b2 = MUL(W5, row[1]);

    MAC(b2, -W1, row[3]);

    b3 = MUL(W7, row[1]);

    MAC(b3, -W5, row[3]);



    if (AV_RN64A(row + 4)) {

        a0 +=   W4*row[4] + W6*row[6];

        a1 += - W4*row[4] - W2*row[6];

        a2 += - W4*row[4] + W2*row[6];

        a3 +=   W4*row[4] - W6*row[6];



        MAC(b0,  W5, row[5]);

        MAC(b0,  W7, row[7]);



        MAC(b1, -W1, row[5]);

        MAC(b1, -W5, row[7]);



        MAC(b2,  W7, row[5]);

        MAC(b2,  W3, row[7]);



        MAC(b3,  W3, row[5]);

        MAC(b3, -W1, row[7]);

    }



    row[0] = (a0 + b0) >> (ROW_SHIFT + extra_shift);

    row[7] = (a0 - b0) >> (ROW_SHIFT + extra_shift);

    row[1] = (a1 + b1) >> (ROW_SHIFT + extra_shift);

    row[6] = (a1 - b1) >> (ROW_SHIFT + extra_shift);

    row[2] = (a2 + b2) >> (ROW_SHIFT + extra_shift);

    row[5] = (a2 - b2) >> (ROW_SHIFT + extra_shift);

    row[3] = (a3 + b3) >> (ROW_SHIFT + extra_shift);

    row[4] = (a3 - b3) >> (ROW_SHIFT + extra_shift);

}
