static void dct_error(const struct algo *dct, int test, int is_idct, int speed)

{

    int it, i, scale;

    int err_inf, v;

    int64_t err2, ti, ti1, it1;

    int64_t sysErr[64], sysErrMax = 0;

    int maxout = 0;

    int blockSumErrMax = 0, blockSumErr;

    AVLFG prng;



    av_lfg_init(&prng, 1);



    err_inf = 0;

    err2 = 0;

    for (i = 0; i < 64; i++)

        sysErr[i] = 0;

    for (it = 0; it < NB_ITS; it++) {

        for (i = 0; i < 64; i++)

            block1[i] = 0;

        switch (test) {

        case 0:

            for (i = 0; i < 64; i++)

                block1[i] = (av_lfg_get(&prng) % 512) - 256;

            if (is_idct) {

                ff_ref_fdct(block1);

                for (i = 0; i < 64; i++)

                    block1[i] >>= 3;

            }

            break;

        case 1: {

                int num = av_lfg_get(&prng) % 10 + 1;

                for (i = 0; i < num; i++)

                    block1[av_lfg_get(&prng) % 64] =

                        av_lfg_get(&prng) % 512 - 256;

            }

            break;

        case 2:

            block1[0] = av_lfg_get(&prng) % 4096 - 2048;

            block1[63] = (block1[0] & 1) ^ 1;

            break;

        }



        for (i = 0; i < 64; i++)

            block_org[i] = block1[i];



        if (dct->format == MMX_PERM) {

            for (i = 0; i < 64; i++)

                block[idct_mmx_perm[i]] = block1[i];

        } else if (dct->format == MMX_SIMPLE_PERM) {

            for (i = 0; i < 64; i++)

                block[idct_simple_mmx_perm[i]] = block1[i];

        } else if (dct->format == SSE2_PERM) {

            for (i = 0; i < 64; i++)

                block[(i & 0x38) | idct_sse2_row_perm[i & 7]] = block1[i];

        } else if (dct->format == PARTTRANS_PERM) {

            for (i = 0; i < 64; i++)

                block[(i & 0x24) | ((i & 3) << 3) | ((i >> 3) & 3)] = block1[i];

        } else {

            for (i = 0; i < 64; i++)

                block[i] = block1[i];

        }



        dct->func(block);

        mmx_emms();



        if (dct->format == SCALE_PERM) {

            for (i = 0; i < 64; i++) {

                scale = 8 * (1 << (AANSCALE_BITS + 11)) / ff_aanscales[i];

                block[i] = (block[i] * scale) >> AANSCALE_BITS;

            }

        }



        dct->ref(block1);



        blockSumErr = 0;

        for (i = 0; i < 64; i++) {

            v = abs(block[i] - block1[i]);

            if (v > err_inf)

                err_inf = v;

            err2 += v * v;

            sysErr[i] += block[i] - block1[i];

            blockSumErr += v;

            if (abs(block[i]) > maxout)

                maxout = abs(block[i]);

        }

        if (blockSumErrMax < blockSumErr)

            blockSumErrMax = blockSumErr;

    }

    for (i = 0; i < 64; i++)

        sysErrMax = FFMAX(sysErrMax, FFABS(sysErr[i]));



    for (i = 0; i < 64; i++) {

        if (i % 8 == 0)

            printf("\n");

        printf("%7d ", (int) sysErr[i]);

    }

    printf("\n");



    printf("%s %s: err_inf=%d err2=%0.8f syserr=%0.8f maxout=%d blockSumErr=%d\n",

           is_idct ? "IDCT" : "DCT", dct->name, err_inf,

           (double) err2 / NB_ITS / 64.0, (double) sysErrMax / NB_ITS,

           maxout, blockSumErrMax);



    if (!speed)

        return;



    /* speed test */

    for (i = 0; i < 64; i++)

        block1[i] = 0;



    switch (test) {

    case 0:

        for (i = 0; i < 64; i++)

            block1[i] = av_lfg_get(&prng) % 512 - 256;

        if (is_idct) {

            ff_ref_fdct(block1);

            for (i = 0; i < 64; i++)

                block1[i] >>= 3;

        }

        break;

    case 1:

    case 2:

        block1[0] = av_lfg_get(&prng) % 512 - 256;

        block1[1] = av_lfg_get(&prng) % 512 - 256;

        block1[2] = av_lfg_get(&prng) % 512 - 256;

        block1[3] = av_lfg_get(&prng) % 512 - 256;

        break;

    }



    if (dct->format == MMX_PERM) {

        for (i = 0; i < 64; i++)

            block[idct_mmx_perm[i]] = block1[i];

    } else if (dct->format == MMX_SIMPLE_PERM) {

        for (i = 0; i < 64; i++)

            block[idct_simple_mmx_perm[i]] = block1[i];

    } else {

        for (i = 0; i < 64; i++)

            block[i] = block1[i];

    }



    ti = gettime();

    it1 = 0;

    do {

        for (it = 0; it < NB_ITS_SPEED; it++) {

            for (i = 0; i < 64; i++)

                block[i] = block1[i];

            dct->func(block);

        }

        it1 += NB_ITS_SPEED;

        ti1 = gettime() - ti;

    } while (ti1 < 1000000);

    mmx_emms();



    printf("%s %s: %0.1f kdct/s\n", is_idct ? "IDCT" : "DCT", dct->name,

           (double) it1 * 1000.0 / (double) ti1);

}
