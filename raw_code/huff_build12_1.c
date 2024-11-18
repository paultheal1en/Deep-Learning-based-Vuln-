static int huff_build12(VLC *vlc, uint8_t *len)

{

    HuffEntry he[4096];

    uint32_t codes[4096];

    uint8_t bits[4096];

    uint16_t syms[4096];

    uint32_t code;

    int i;



    for (i = 0; i < 4096; i++) {

        he[i].sym = 4095 - i;

        he[i].len = len[i];

        if (len[i] == 0)

            return AVERROR_INVALIDDATA;

    }

    AV_QSORT(he, 4096, HuffEntry, huff_cmp_len12);



    code = 1;

    for (i = 4095; i >= 0; i--) {

        codes[i] = code >> (32 - he[i].len);

        bits[i]  = he[i].len;

        syms[i]  = he[i].sym;

        code += 0x80000000u >> (he[i].len - 1);

    }



    ff_free_vlc(vlc);

    return ff_init_vlc_sparse(vlc, FFMIN(he[4095].len, 14), 4096,

                              bits,  sizeof(*bits),  sizeof(*bits),

                              codes, sizeof(*codes), sizeof(*codes),

                              syms,  sizeof(*syms),  sizeof(*syms), 0);

}
