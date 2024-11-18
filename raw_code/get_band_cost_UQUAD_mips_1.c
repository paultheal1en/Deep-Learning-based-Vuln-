static float get_band_cost_UQUAD_mips(struct AACEncContext *s,

                                      PutBitContext *pb, const float *in,

                                      const float *scaled, int size, int scale_idx,

                                      int cb, const float lambda, const float uplim,

                                      int *bits)

{

    const float Q34 = ff_aac_pow34sf_tab[POW_SF2_ZERO - scale_idx + SCALE_ONE_POS - SCALE_DIV_512];

    const float IQ  = ff_aac_pow2sf_tab [POW_SF2_ZERO + scale_idx - SCALE_ONE_POS + SCALE_DIV_512];

    int i;

    float cost = 0;

    int curbits = 0;

    int qc1, qc2, qc3, qc4;



    uint8_t *p_bits  = (uint8_t*)ff_aac_spectral_bits[cb-1];

    float   *p_codes = (float  *)ff_aac_codebook_vectors[cb-1];



    for (i = 0; i < size; i += 4) {

        const float *vec;

        int curidx;

        float *in_pos = (float *)&in[i];

        float di0, di1, di2, di3;

        int t0, t1, t2, t3, t4;



        qc1 = scaled[i  ] * Q34 + ROUND_STANDARD;

        qc2 = scaled[i+1] * Q34 + ROUND_STANDARD;

        qc3 = scaled[i+2] * Q34 + ROUND_STANDARD;

        qc4 = scaled[i+3] * Q34 + ROUND_STANDARD;



        __asm__ volatile (

            ".set push                                  \n\t"

            ".set noreorder                             \n\t"



            "ori        %[t4],  $zero,  2               \n\t"

            "slt        %[t0],  %[t4],  %[qc1]          \n\t"

            "slt        %[t1],  %[t4],  %[qc2]          \n\t"

            "slt        %[t2],  %[t4],  %[qc3]          \n\t"

            "slt        %[t3],  %[t4],  %[qc4]          \n\t"

            "movn       %[qc1], %[t4],  %[t0]           \n\t"

            "movn       %[qc2], %[t4],  %[t1]           \n\t"

            "movn       %[qc3], %[t4],  %[t2]           \n\t"

            "movn       %[qc4], %[t4],  %[t3]           \n\t"



            ".set pop                                   \n\t"



            : [qc1]"+r"(qc1), [qc2]"+r"(qc2),

              [qc3]"+r"(qc3), [qc4]"+r"(qc4),

              [t0]"=&r"(t0), [t1]"=&r"(t1), [t2]"=&r"(t2), [t3]"=&r"(t3),

              [t4]"=&r"(t4)

        );



        curidx = qc1;

        curidx *= 3;

        curidx += qc2;

        curidx *= 3;

        curidx += qc3;

        curidx *= 3;

        curidx += qc4;



        curbits += p_bits[curidx];

        curbits += uquad_sign_bits[curidx];

        vec     = &p_codes[curidx*4];



        __asm__ volatile (

            ".set push                                  \n\t"

            ".set noreorder                             \n\t"



            "lwc1       %[di0], 0(%[in_pos])            \n\t"

            "lwc1       %[di1], 4(%[in_pos])            \n\t"

            "lwc1       %[di2], 8(%[in_pos])            \n\t"

            "lwc1       %[di3], 12(%[in_pos])           \n\t"

            "abs.s      %[di0], %[di0]                  \n\t"

            "abs.s      %[di1], %[di1]                  \n\t"

            "abs.s      %[di2], %[di2]                  \n\t"

            "abs.s      %[di3], %[di3]                  \n\t"

            "lwc1       $f0,    0(%[vec])               \n\t"

            "lwc1       $f1,    4(%[vec])               \n\t"

            "lwc1       $f2,    8(%[vec])               \n\t"

            "lwc1       $f3,    12(%[vec])              \n\t"

            "nmsub.s    %[di0], %[di0], $f0,    %[IQ]   \n\t"

            "nmsub.s    %[di1], %[di1], $f1,    %[IQ]   \n\t"

            "nmsub.s    %[di2], %[di2], $f2,    %[IQ]   \n\t"

            "nmsub.s    %[di3], %[di3], $f3,    %[IQ]   \n\t"



            ".set pop                                   \n\t"



            : [di0]"=&f"(di0), [di1]"=&f"(di1),

              [di2]"=&f"(di2), [di3]"=&f"(di3)

            : [in_pos]"r"(in_pos), [vec]"r"(vec),

              [IQ]"f"(IQ)

            : "$f0", "$f1", "$f2", "$f3",

              "memory"

        );



        cost += di0 * di0 + di1 * di1

                + di2 * di2 + di3 * di3;

    }



    if (bits)

        *bits = curbits;

    return cost * lambda + curbits;

}
