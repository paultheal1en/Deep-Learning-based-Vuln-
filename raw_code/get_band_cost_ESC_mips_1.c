static float get_band_cost_ESC_mips(struct AACEncContext *s,

                                    PutBitContext *pb, const float *in,

                                    const float *scaled, int size, int scale_idx,

                                    int cb, const float lambda, const float uplim,

                                    int *bits)

{

    const float Q34 = ff_aac_pow34sf_tab[POW_SF2_ZERO - scale_idx + SCALE_ONE_POS - SCALE_DIV_512];

    const float IQ  = ff_aac_pow2sf_tab [POW_SF2_ZERO + scale_idx - SCALE_ONE_POS + SCALE_DIV_512];

    const float CLIPPED_ESCAPE = 165140.0f * IQ;

    int i;

    float cost = 0;

    int qc1, qc2, qc3, qc4;

    int curbits = 0;



    uint8_t *p_bits  = (uint8_t*)ff_aac_spectral_bits[cb-1];

    float   *p_codes = (float*  )ff_aac_codebook_vectors[cb-1];



    for (i = 0; i < size; i += 4) {

        const float *vec, *vec2;

        int curidx, curidx2;

        float t1, t2, t3, t4;

        float di1, di2, di3, di4;

        int cond0, cond1, cond2, cond3;

        int c1, c2, c3, c4;

        int t6, t7;



        qc1 = scaled[i  ] * Q34 + ROUND_STANDARD;

        qc2 = scaled[i+1] * Q34 + ROUND_STANDARD;

        qc3 = scaled[i+2] * Q34 + ROUND_STANDARD;

        qc4 = scaled[i+3] * Q34 + ROUND_STANDARD;



        __asm__ volatile (

            ".set push                                  \n\t"

            ".set noreorder                             \n\t"



            "ori        %[t6],      $zero,  15          \n\t"

            "ori        %[t7],      $zero,  16          \n\t"

            "shll_s.w   %[c1],      %[qc1], 18          \n\t"

            "shll_s.w   %[c2],      %[qc2], 18          \n\t"

            "shll_s.w   %[c3],      %[qc3], 18          \n\t"

            "shll_s.w   %[c4],      %[qc4], 18          \n\t"

            "srl        %[c1],      %[c1],  18          \n\t"

            "srl        %[c2],      %[c2],  18          \n\t"

            "srl        %[c3],      %[c3],  18          \n\t"

            "srl        %[c4],      %[c4],  18          \n\t"

            "slt        %[cond0],   %[t6],  %[qc1]      \n\t"

            "slt        %[cond1],   %[t6],  %[qc2]      \n\t"

            "slt        %[cond2],   %[t6],  %[qc3]      \n\t"

            "slt        %[cond3],   %[t6],  %[qc4]      \n\t"

            "movn       %[qc1],     %[t7],  %[cond0]    \n\t"

            "movn       %[qc2],     %[t7],  %[cond1]    \n\t"

            "movn       %[qc3],     %[t7],  %[cond2]    \n\t"

            "movn       %[qc4],     %[t7],  %[cond3]    \n\t"



            ".set pop                                   \n\t"



            : [qc1]"+r"(qc1), [qc2]"+r"(qc2),

              [qc3]"+r"(qc3), [qc4]"+r"(qc4),

              [cond0]"=&r"(cond0), [cond1]"=&r"(cond1),

              [cond2]"=&r"(cond2), [cond3]"=&r"(cond3),

              [c1]"=&r"(c1), [c2]"=&r"(c2),

              [c3]"=&r"(c3), [c4]"=&r"(c4),

              [t6]"=&r"(t6), [t7]"=&r"(t7)

        );



        curidx = 17 * qc1;

        curidx += qc2;



        curidx2 = 17 * qc3;

        curidx2 += qc4;



        curbits += p_bits[curidx];

        curbits += esc_sign_bits[curidx];

        vec     = &p_codes[curidx*2];



        curbits += p_bits[curidx2];

        curbits += esc_sign_bits[curidx2];

        vec2     = &p_codes[curidx2*2];



        curbits += (av_log2(c1) * 2 - 3) & (-cond0);

        curbits += (av_log2(c2) * 2 - 3) & (-cond1);

        curbits += (av_log2(c3) * 2 - 3) & (-cond2);

        curbits += (av_log2(c4) * 2 - 3) & (-cond3);



        t1 = fabsf(in[i  ]);

        t2 = fabsf(in[i+1]);

        t3 = fabsf(in[i+2]);

        t4 = fabsf(in[i+3]);



        if (cond0) {

            if (t1 >= CLIPPED_ESCAPE) {

                di1 = t1 - CLIPPED_ESCAPE;

            } else {

                di1 = t1 - c1 * cbrtf(c1) * IQ;

            }

        } else

            di1 = t1 - vec[0] * IQ;



        if (cond1) {

            if (t2 >= CLIPPED_ESCAPE) {

                di2 = t2 - CLIPPED_ESCAPE;

            } else {

                di2 = t2 - c2 * cbrtf(c2) * IQ;

            }

        } else

            di2 = t2 - vec[1] * IQ;



        if (cond2) {

            if (t3 >= CLIPPED_ESCAPE) {

                di3 = t3 - CLIPPED_ESCAPE;

            } else {

                di3 = t3 - c3 * cbrtf(c3) * IQ;

            }

        } else

            di3 = t3 - vec2[0] * IQ;



        if (cond3) {

            if (t4 >= CLIPPED_ESCAPE) {

                di4 = t4 - CLIPPED_ESCAPE;

            } else {

                di4 = t4 - c4 * cbrtf(c4) * IQ;

            }

        } else

            di4 = t4 - vec2[1]*IQ;



        cost += di1 * di1 + di2 * di2

                + di3 * di3 + di4 * di4;

    }



    if (bits)

        *bits = curbits;

    return cost * lambda + curbits;

}
