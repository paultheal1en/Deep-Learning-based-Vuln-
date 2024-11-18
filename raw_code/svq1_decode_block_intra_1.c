static int svq1_decode_block_intra(GetBitContext *bitbuf, uint8_t *pixels,

                                   ptrdiff_t pitch)

{

    uint32_t bit_cache;

    uint8_t *list[63];

    uint32_t *dst;

    const uint32_t *codebook;

    int entries[6];

    int i, j, m, n;

    int stages;

    unsigned mean;

    unsigned x, y, width, height, level;

    uint32_t n1, n2, n3, n4;



    /* initialize list for breadth first processing of vectors */

    list[0] = pixels;



    /* recursively process vector */

    for (i = 0, m = 1, n = 1, level = 5; i < n; i++) {

        SVQ1_PROCESS_VECTOR();



        /* destination address and vector size */

        dst    = (uint32_t *)list[i];

        width  = 1 << ((4 + level) / 2);

        height = 1 << ((3 + level) / 2);



        /* get number of stages (-1 skips vector, 0 for mean only) */

        stages = get_vlc2(bitbuf, svq1_intra_multistage[level].table, 3, 3) - 1;



        if (stages == -1) {

            for (y = 0; y < height; y++)

                memset(&dst[y * (pitch / 4)], 0, width);

            continue;   /* skip vector */

        }



        if ((stages > 0 && level >= 4)) {

            ff_dlog(NULL,

                    "Error (svq1_decode_block_intra): invalid vector: stages=%i level=%i\n",

                    stages, level);

            return AVERROR_INVALIDDATA;  /* invalid vector */

        }

        av_assert0(stages >= 0);



        mean = get_vlc2(bitbuf, svq1_intra_mean.table, 8, 3);



        if (stages == 0) {

            for (y = 0; y < height; y++)

                memset(&dst[y * (pitch / 4)], mean, width);

        } else {

            SVQ1_CALC_CODEBOOK_ENTRIES(ff_svq1_intra_codebooks);



            for (y = 0; y < height; y++) {

                for (x = 0; x < width / 4; x++, codebook++) {

                    n1 = n4;

                    n2 = n4;

                    SVQ1_ADD_CODEBOOK()

                    /* store result */

                    dst[x] = n1 << 8 | n2;

                }

                dst += pitch / 4;

            }

        }

    }



    return 0;

}
