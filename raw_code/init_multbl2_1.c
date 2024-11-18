static void init_multbl2(uint8_t tbl[1024], const int c[4],

                         const uint8_t *log8, const uint8_t *alog8,

                         const uint8_t *sbox)

{

    int i, j;



    for (i = 0; i < 1024; i++) {

        int x = sbox[i >> 2];

        if (x)

            tbl[i] = alog8[log8[x] + log8[c[i & 3]]];

    }

#if !CONFIG_SMALL

    for (j = 256; j < 1024; j++)

        for (i = 0; i < 4; i++)

            tbl[4*j + i] = tbl[4*j + ((i - 1) & 3) - 1024];

#endif

}
