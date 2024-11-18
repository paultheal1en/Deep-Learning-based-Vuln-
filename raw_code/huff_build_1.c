static int huff_build(VLC *vlc, uint8_t *len)

{

    HuffEntry he[256];

    uint32_t codes[256];

    uint8_t bits[256];

    uint8_t syms[256];

    uint32_t code;

    int i;



    for (i = 0; i < 256; i++) {

        he[i].sym = 255 - i;

        he[i].len = len[i];

        if (len[i] == 0)

            return AVERROR_INVALIDDATA;

    }

    AV_QSORT(he, 256, HuffEntry, huff_cmp_len);



    code = 1;

    for (i = 255; i >= 0; i--) {

        codes[i] = code >> (32 - he[i].len);

        bits[i]  = he[i].len;

        syms[i]  = he[i].sym;

        code += 0x80000000u >> (he[i].len - 1);

    }



    ff_free_vlc(vlc);

    return ff_init_vlc_sparse(vlc, FFMIN(he[255].len, 12), 256,

                              bits,  sizeof(*bits),  sizeof(*bits),

                              codes, sizeof(*codes), sizeof(*codes),

                              syms,  sizeof(*syms),  sizeof(*syms), 0);

}
