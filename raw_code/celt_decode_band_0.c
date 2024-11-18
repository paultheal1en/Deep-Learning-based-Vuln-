static unsigned int celt_decode_band(CeltContext *s, OpusRangeCoder *rc,

                                     const int band, float *X, float *Y,

                                     int N, int b, unsigned int blocks,

                                     float *lowband, int duration,

                                     float *lowband_out, int level,

                                     float gain, float *lowband_scratch,

                                     int fill)

{

    const uint8_t *cache;

    int dualstereo, split;

    int imid = 0, iside = 0;

    unsigned int N0 = N;

    int N_B;

    int N_B0;

    int B0 = blocks;

    int time_divide = 0;

    int recombine = 0;

    int inv = 0;

    float mid = 0, side = 0;

    int longblocks = (B0 == 1);

    unsigned int cm = 0;



    N_B0 = N_B = N / blocks;

    split = dualstereo = (Y != NULL);



    if (N == 1) {

        /* special case for one sample */

        int i;

        float *x = X;

        for (i = 0; i <= dualstereo; i++) {

            int sign = 0;

            if (s->remaining2 >= 1<<3) {

                sign           = opus_getrawbits(rc, 1);

                s->remaining2 -= 1 << 3;

                b             -= 1 << 3;

            }

            x[0] = sign ? -1.0f : 1.0f;

            x = Y;

        }

        if (lowband_out)

            lowband_out[0] = X[0];

        return 1;

    }



    if (!dualstereo && level == 0) {

        int tf_change = s->tf_change[band];

        int k;

        if (tf_change > 0)

            recombine = tf_change;

        /* Band recombining to increase frequency resolution */



        if (lowband &&

            (recombine || ((N_B & 1) == 0 && tf_change < 0) || B0 > 1)) {

            int j;

            for (j = 0; j < N; j++)

                lowband_scratch[j] = lowband[j];

            lowband = lowband_scratch;

        }



        for (k = 0; k < recombine; k++) {

            if (lowband)

                celt_haar1(lowband, N >> k, 1 << k);

            fill = celt_bit_interleave[fill & 0xF] | celt_bit_interleave[fill >> 4] << 2;

        }

        blocks >>= recombine;

        N_B <<= recombine;



        /* Increasing the time resolution */

        while ((N_B & 1) == 0 && tf_change < 0) {

            if (lowband)

                celt_haar1(lowband, N_B, blocks);

            fill |= fill << blocks;

            blocks <<= 1;

            N_B >>= 1;

            time_divide++;

            tf_change++;

        }

        B0 = blocks;

        N_B0 = N_B;



        /* Reorganize the samples in time order instead of frequency order */

        if (B0 > 1 && lowband)

            celt_deinterleave_hadamard(s->scratch, lowband, N_B >> recombine,

                                       B0 << recombine, longblocks);

    }



    /* If we need 1.5 more bit than we can produce, split the band in two. */

    cache = celt_cache_bits +

            celt_cache_index[(duration + 1) * CELT_MAX_BANDS + band];

    if (!dualstereo && duration >= 0 && b > cache[cache[0]] + 12 && N > 2) {

        N >>= 1;

        Y = X + N;

        split = 1;

        duration -= 1;

        if (blocks == 1)

            fill = (fill & 1) | (fill << 1);

        blocks = (blocks + 1) >> 1;

    }



    if (split) {

        int qn;

        int itheta = 0;

        int mbits, sbits, delta;

        int qalloc;

        int pulse_cap;

        int offset;

        int orig_fill;

        int tell;



        /* Decide on the resolution to give to the split parameter theta */

        pulse_cap = celt_log_freq_range[band] + duration * 8;

        offset = (pulse_cap >> 1) - (dualstereo && N == 2 ? CELT_QTHETA_OFFSET_TWOPHASE :

                                                          CELT_QTHETA_OFFSET);

        qn = (dualstereo && band >= s->intensitystereo) ? 1 :

             celt_compute_qn(N, b, offset, pulse_cap, dualstereo);

        tell = opus_rc_tell_frac(rc);

        if (qn != 1) {

            /* Entropy coding of the angle. We use a uniform pdf for the

            time split, a step for stereo, and a triangular one for the rest. */

            if (dualstereo && N > 2)

                itheta = opus_rc_stepmodel(rc, qn/2);

            else if (dualstereo || B0 > 1)

                itheta = opus_rc_unimodel(rc, qn+1);

            else

                itheta = opus_rc_trimodel(rc, qn);

            itheta = itheta * 16384 / qn;

            /* NOTE: Renormalising X and Y *may* help fixed-point a bit at very high rate.

            Let's do that at higher complexity */

        } else if (dualstereo) {

            inv = (b > 2 << 3 && s->remaining2 > 2 << 3) ? opus_rc_p2model(rc, 2) : 0;

            itheta = 0;

        }

        qalloc = opus_rc_tell_frac(rc) - tell;

        b -= qalloc;



        orig_fill = fill;

        if (itheta == 0) {

            imid = 32767;

            iside = 0;

            fill &= (1 << blocks) - 1;

            delta = -16384;

        } else if (itheta == 16384) {

            imid = 0;

            iside = 32767;

            fill &= ((1 << blocks) - 1) << blocks;

            delta = 16384;

        } else {

            imid = celt_cos(itheta);

            iside = celt_cos(16384-itheta);

            /* This is the mid vs side allocation that minimizes squared error

            in that band. */

            delta = ROUND_MUL16((N - 1) << 7, celt_log2tan(iside, imid));

        }



        mid  = imid  / 32768.0f;

        side = iside / 32768.0f;



        /* This is a special case for N=2 that only works for stereo and takes

        advantage of the fact that mid and side are orthogonal to encode

        the side with just one bit. */

        if (N == 2 && dualstereo) {

            int c;

            int sign = 0;

            float tmp;

            float *x2, *y2;

            mbits = b;

            /* Only need one bit for the side */

            sbits = (itheta != 0 && itheta != 16384) ? 1 << 3 : 0;

            mbits -= sbits;

            c = (itheta > 8192);

            s->remaining2 -= qalloc+sbits;



            x2 = c ? Y : X;

            y2 = c ? X : Y;

            if (sbits)

                sign = opus_getrawbits(rc, 1);

            sign = 1 - 2 * sign;

            /* We use orig_fill here because we want to fold the side, but if

            itheta==16384, we'll have cleared the low bits of fill. */

            cm = celt_decode_band(s, rc, band, x2, NULL, N, mbits, blocks,

                                  lowband, duration, lowband_out, level, gain,

                                  lowband_scratch, orig_fill);

            /* We don't split N=2 bands, so cm is either 1 or 0 (for a fold-collapse),

            and there's no need to worry about mixing with the other channel. */

            y2[0] = -sign * x2[1];

            y2[1] =  sign * x2[0];

            X[0] *= mid;

            X[1] *= mid;

            Y[0] *= side;

            Y[1] *= side;

            tmp = X[0];

            X[0] = tmp - Y[0];

            Y[0] = tmp + Y[0];

            tmp = X[1];

            X[1] = tmp - Y[1];

            Y[1] = tmp + Y[1];

        } else {

            /* "Normal" split code */

            float *next_lowband2     = NULL;

            float *next_lowband_out1 = NULL;

            int next_level = 0;

            int rebalance;



            /* Give more bits to low-energy MDCTs than they would

             * otherwise deserve */

            if (B0 > 1 && !dualstereo && (itheta & 0x3fff)) {

                if (itheta > 8192)

                    /* Rough approximation for pre-echo masking */

                    delta -= delta >> (4 - duration);

                else

                    /* Corresponds to a forward-masking slope of

                     * 1.5 dB per 10 ms */

                    delta = FFMIN(0, delta + (N << 3 >> (5 - duration)));

            }

            mbits = av_clip((b - delta) / 2, 0, b);

            sbits = b - mbits;

            s->remaining2 -= qalloc;



            if (lowband && !dualstereo)

                next_lowband2 = lowband + N; /* >32-bit split case */



            /* Only stereo needs to pass on lowband_out.

             * Otherwise, it's handled at the end */

            if (dualstereo)

                next_lowband_out1 = lowband_out;

            else

                next_level = level + 1;



            rebalance = s->remaining2;

            if (mbits >= sbits) {

                /* In stereo mode, we do not apply a scaling to the mid

                 * because we need the normalized mid for folding later */

                cm = celt_decode_band(s, rc, band, X, NULL, N, mbits, blocks,

                                      lowband, duration, next_lowband_out1,

                                      next_level, dualstereo ? 1.0f : (gain * mid),

                                      lowband_scratch, fill);



                rebalance = mbits - (rebalance - s->remaining2);

                if (rebalance > 3 << 3 && itheta != 0)

                    sbits += rebalance - (3 << 3);



                /* For a stereo split, the high bits of fill are always zero,

                 * so no folding will be done to the side. */

                cm |= celt_decode_band(s, rc, band, Y, NULL, N, sbits, blocks,

                                       next_lowband2, duration, NULL,

                                       next_level, gain * side, NULL,

                                       fill >> blocks) << ((B0 >> 1) & (dualstereo - 1));

            } else {

                /* For a stereo split, the high bits of fill are always zero,

                 * so no folding will be done to the side. */

                cm = celt_decode_band(s, rc, band, Y, NULL, N, sbits, blocks,

                                      next_lowband2, duration, NULL,

                                      next_level, gain * side, NULL,

                                      fill >> blocks) << ((B0 >> 1) & (dualstereo - 1));



                rebalance = sbits - (rebalance - s->remaining2);

                if (rebalance > 3 << 3 && itheta != 16384)

                    mbits += rebalance - (3 << 3);



                /* In stereo mode, we do not apply a scaling to the mid because

                 * we need the normalized mid for folding later */

                cm |= celt_decode_band(s, rc, band, X, NULL, N, mbits, blocks,

                                       lowband, duration, next_lowband_out1,

                                       next_level, dualstereo ? 1.0f : (gain * mid),

                                       lowband_scratch, fill);

            }

        }

    } else {

        /* This is the basic no-split case */

        unsigned int q         = celt_bits2pulses(cache, b);

        unsigned int curr_bits = celt_pulses2bits(cache, q);

        s->remaining2 -= curr_bits;



        /* Ensures we can never bust the budget */

        while (s->remaining2 < 0 && q > 0) {

            s->remaining2 += curr_bits;

            curr_bits      = celt_pulses2bits(cache, --q);

            s->remaining2 -= curr_bits;

        }



        if (q != 0) {

            /* Finally do the actual quantization */

            cm = celt_alg_unquant(rc, X, N, (q < 8) ? q : (8 + (q & 7)) << ((q >> 3) - 1),

                                  s->spread, blocks, gain);

        } else {

            /* If there's no pulse, fill the band anyway */

            int j;

            unsigned int cm_mask = (1 << blocks) - 1;

            fill &= cm_mask;

            if (!fill) {

                for (j = 0; j < N; j++)

                    X[j] = 0.0f;

            } else {

                if (lowband == NULL) {

                    /* Noise */

                    for (j = 0; j < N; j++)

                        X[j] = (((int32_t)celt_rng(s)) >> 20);

                    cm = cm_mask;

                } else {

                    /* Folded spectrum */

                    for (j = 0; j < N; j++) {

                        /* About 48 dB below the "normal" folding level */

                        X[j] = lowband[j] + (((celt_rng(s)) & 0x8000) ? 1.0f / 256 : -1.0f / 256);

                    }

                    cm = fill;

                }

                celt_renormalize_vector(X, N, gain);

            }

        }

    }



    /* This code is used by the decoder and by the resynthesis-enabled encoder */

    if (dualstereo) {

        int j;

        if (N != 2)

            celt_stereo_merge(X, Y, mid, N);

        if (inv) {

            for (j = 0; j < N; j++)

                Y[j] *= -1;

        }

    } else if (level == 0) {

        int k;



        /* Undo the sample reorganization going from time order to frequency order */

        if (B0 > 1)

            celt_interleave_hadamard(s->scratch, X, N_B>>recombine,

                                     B0<<recombine, longblocks);



        /* Undo time-freq changes that we did earlier */

        N_B = N_B0;

        blocks = B0;

        for (k = 0; k < time_divide; k++) {

            blocks >>= 1;

            N_B <<= 1;

            cm |= cm >> blocks;

            celt_haar1(X, N_B, blocks);

        }



        for (k = 0; k < recombine; k++) {

            cm = celt_bit_deinterleave[cm];

            celt_haar1(X, N0>>k, 1<<k);

        }

        blocks <<= recombine;



        /* Scale output for later folding */

        if (lowband_out) {

            int j;

            float n = sqrtf(N0);

            for (j = 0; j < N0; j++)

                lowband_out[j] = n * X[j];

        }

        cm &= (1 << blocks) - 1;

    }

    return cm;

}
