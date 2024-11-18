uint32_t ff_celt_encode_band(CeltFrame *f, OpusRangeCoder *rc, const int band,

                             float *X, float *Y, int N, int b, uint32_t blocks,

                             float *lowband, int duration, float *lowband_out, int level,

                             float gain, float *lowband_scratch, int fill)

{

    const uint8_t *cache;

    int dualstereo, split;

    int imid = 0, iside = 0;

    //uint32_t N0 = N;

    int N_B;

    //int N_B0;

    int B0 = blocks;

    int time_divide = 0;

    int recombine = 0;

    int inv = 0;

    float mid = 0, side = 0;

    int longblocks = (B0 == 1);

    uint32_t cm = 0;



    //N_B0 = N_B = N / blocks;

    split = dualstereo = (Y != NULL);



    if (N == 1) {

        /* special case for one sample - the decoder's output will be +- 1.0f!!! */

        int i;

        float *x = X;

        for (i = 0; i <= dualstereo; i++) {

            if (f->remaining2 >= 1<<3) {

                ff_opus_rc_put_raw(rc, x[0] < 0, 1);

                f->remaining2 -= 1 << 3;

                b             -= 1 << 3;

            }

            x = Y;

        }

        if (lowband_out)

            lowband_out[0] = X[0];

        return 1;

    }



    if (!dualstereo && level == 0) {

        int tf_change = f->tf_change[band];

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

            celt_haar1(X, N >> k, 1 << k);

            fill = ff_celt_bit_interleave[fill & 0xF] | ff_celt_bit_interleave[fill >> 4] << 2;

        }

        blocks >>= recombine;

        N_B <<= recombine;



        /* Increasing the time resolution */

        while ((N_B & 1) == 0 && tf_change < 0) {

            celt_haar1(X, N_B, blocks);

            fill |= fill << blocks;

            blocks <<= 1;

            N_B >>= 1;

            time_divide++;

            tf_change++;

        }

        B0 = blocks;

        //N_B0 = N_B;



        /* Reorganize the samples in time order instead of frequency order */

        if (B0 > 1)

            celt_deinterleave_hadamard(f->scratch, X, N_B >> recombine,

                                       B0 << recombine, longblocks);

    }



    /* If we need 1.5 more bit than we can produce, split the band in two. */

    cache = ff_celt_cache_bits +

            ff_celt_cache_index[(duration + 1) * CELT_MAX_BANDS + band];

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

        int itheta = celt_calc_theta(X, Y, dualstereo, N);

        int mbits, sbits, delta;

        int qalloc;

        int pulse_cap;

        int offset;

        int orig_fill;

        int tell;



        /* Decide on the resolution to give to the split parameter theta */

        pulse_cap = ff_celt_log_freq_range[band] + duration * 8;

        offset = (pulse_cap >> 1) - (dualstereo && N == 2 ? CELT_QTHETA_OFFSET_TWOPHASE :

                                                          CELT_QTHETA_OFFSET);

        qn = (dualstereo && band >= f->intensity_stereo) ? 1 :

             celt_compute_qn(N, b, offset, pulse_cap, dualstereo);

        tell = opus_rc_tell_frac(rc);



        if (qn != 1) {



            itheta = (itheta*qn + 8192) >> 14;



            /* Entropy coding of the angle. We use a uniform pdf for the

             * time split, a step for stereo, and a triangular one for the rest. */

            if (dualstereo && N > 2)

                ff_opus_rc_enc_uint_step(rc, itheta, qn / 2);

            else if (dualstereo || B0 > 1)

                ff_opus_rc_enc_uint(rc, itheta, qn + 1);

            else

                ff_opus_rc_enc_uint_tri(rc, itheta, qn);

            itheta = itheta * 16384 / qn;



            if (dualstereo) {

                if (itheta == 0)

                    celt_stereo_is_decouple(X, Y, f->block[0].lin_energy[band], f->block[1].lin_energy[band], N);

                else

                    celt_stereo_ms_decouple(X, Y, N);

            }

        } else if (dualstereo) {

             inv = itheta > 8192;

             if (inv)

             {

                int j;

                for (j=0;j<N;j++)

                   Y[j] = -Y[j];

             }

             celt_stereo_is_decouple(X, Y, f->block[0].lin_energy[band], f->block[1].lin_energy[band], N);



            if (b > 2 << 3 && f->remaining2 > 2 << 3) {

                ff_opus_rc_enc_log(rc, inv, 2);

            } else {

                inv = 0;

            }



            itheta = 0;

        }

        qalloc = opus_rc_tell_frac(rc) - tell;

        b -= qalloc;



        orig_fill = fill;

        if (itheta == 0) {

            imid = 32767;

            iside = 0;

            fill = av_mod_uintp2(fill, blocks);

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

            f->remaining2 -= qalloc+sbits;



            x2 = c ? Y : X;

            y2 = c ? X : Y;

            if (sbits) {

                sign = x2[0]*y2[1] - x2[1]*y2[0] < 0;

                ff_opus_rc_put_raw(rc, sign, 1);

            }

            sign = 1 - 2 * sign;

            /* We use orig_fill here because we want to fold the side, but if

            itheta==16384, we'll have cleared the low bits of fill. */

            cm = ff_celt_encode_band(f, rc, band, x2, NULL, N, mbits, blocks,

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

            f->remaining2 -= qalloc;



            if (lowband && !dualstereo)

                next_lowband2 = lowband + N; /* >32-bit split case */



            /* Only stereo needs to pass on lowband_out.

             * Otherwise, it's handled at the end */

            if (dualstereo)

                next_lowband_out1 = lowband_out;

            else

                next_level = level + 1;



            rebalance = f->remaining2;

            if (mbits >= sbits) {

                /* In stereo mode, we do not apply a scaling to the mid

                 * because we need the normalized mid for folding later */

                cm = ff_celt_encode_band(f, rc, band, X, NULL, N, mbits, blocks,

                                         lowband, duration, next_lowband_out1,

                                         next_level, dualstereo ? 1.0f : (gain * mid),

                                         lowband_scratch, fill);



                rebalance = mbits - (rebalance - f->remaining2);

                if (rebalance > 3 << 3 && itheta != 0)

                    sbits += rebalance - (3 << 3);



                /* For a stereo split, the high bits of fill are always zero,

                 * so no folding will be done to the side. */

                cm |= ff_celt_encode_band(f, rc, band, Y, NULL, N, sbits, blocks,

                                          next_lowband2, duration, NULL,

                                          next_level, gain * side, NULL,

                                          fill >> blocks) << ((B0 >> 1) & (dualstereo - 1));

            } else {

                /* For a stereo split, the high bits of fill are always zero,

                 * so no folding will be done to the side. */

                cm = ff_celt_encode_band(f, rc, band, Y, NULL, N, sbits, blocks,

                                         next_lowband2, duration, NULL,

                                         next_level, gain * side, NULL,

                                         fill >> blocks) << ((B0 >> 1) & (dualstereo - 1));



                rebalance = sbits - (rebalance - f->remaining2);

                if (rebalance > 3 << 3 && itheta != 16384)

                    mbits += rebalance - (3 << 3);



                /* In stereo mode, we do not apply a scaling to the mid because

                 * we need the normalized mid for folding later */

                cm |= ff_celt_encode_band(f, rc, band, X, NULL, N, mbits, blocks,

                                          lowband, duration, next_lowband_out1,

                                          next_level, dualstereo ? 1.0f : (gain * mid),

                                          lowband_scratch, fill);

            }

        }

    } else {

        /* This is the basic no-split case */

        uint32_t q         = celt_bits2pulses(cache, b);

        uint32_t curr_bits = celt_pulses2bits(cache, q);

        f->remaining2 -= curr_bits;



        /* Ensures we can never bust the budget */

        while (f->remaining2 < 0 && q > 0) {

            f->remaining2 += curr_bits;

            curr_bits      = celt_pulses2bits(cache, --q);

            f->remaining2 -= curr_bits;

        }



        if (q != 0) {

            /* Finally do the actual quantization */

            cm = celt_alg_quant(rc, X, N, (q < 8) ? q : (8 + (q & 7)) << ((q >> 3) - 1),

                                f->spread, blocks, gain);

        }

    }



    return cm;

}
