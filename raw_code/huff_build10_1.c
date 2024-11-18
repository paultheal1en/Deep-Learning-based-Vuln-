static int huff_build10(VLC *vlc, uint8_t *len)

{

    HuffEntry he[1024];

    uint32_t codes[1024];

    uint8_t bits[1024];

    uint16_t syms[1024];

    uint32_t code;

    int i;



    for (i = 0; i < 1024; i++) {

        he[i].sym = 1023 - i;

        he[i].len = len[i];



    }

    AV_QSORT(he, 1024, HuffEntry, huff_cmp_len10);



    code = 1;

    for (i = 1023; i >= 0; i--) {

        codes[i] = code >> (32 - he[i].len);

        bits[i]  = he[i].len;

        syms[i]  = he[i].sym;

        code += 0x80000000u >> (he[i].len - 1);

    }



    ff_free_vlc(vlc);

    return ff_init_vlc_sparse(vlc, FFMIN(he[1023].len, 12), 1024,

                              bits,  sizeof(*bits),  sizeof(*bits),

                              codes, sizeof(*codes), sizeof(*codes),

                              syms,  sizeof(*syms),  sizeof(*syms), 0);

}