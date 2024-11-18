static void ff_acelp_interpolatef_mips(float *out, const float *in,

                           const float *filter_coeffs, int precision,

                           int frac_pos, int filter_length, int length)

{

    int n, i;

    int prec = precision * 4;

    int fc_offset = precision - frac_pos;

    float in_val_p, in_val_m, fc_val_p, fc_val_m;



    for (n = 0; n < length; n++) {

        /**

        * four pointers are defined in order to minimize number of

        * computations done in inner loop

        */

        const float *p_in_p = &in[n];

        const float *p_in_m = &in[n-1];

        const float *p_filter_coeffs_p = &filter_coeffs[frac_pos];

        const float *p_filter_coeffs_m = filter_coeffs + fc_offset;

        float v = 0;



        for (i = 0; i < filter_length;i++) {

            __asm__ volatile (

                "lwc1   %[in_val_p],           0(%[p_in_p])                    \n\t"

                "lwc1   %[fc_val_p],           0(%[p_filter_coeffs_p])         \n\t"

                "lwc1   %[in_val_m],           0(%[p_in_m])                    \n\t"

                "lwc1   %[fc_val_m],           0(%[p_filter_coeffs_m])         \n\t"

                "addiu  %[p_in_p],             %[p_in_p],              4       \n\t"

                "madd.s %[v],%[v],             %[in_val_p],%[fc_val_p]         \n\t"

                "addiu  %[p_in_m],             %[p_in_m],              -4      \n\t"

                "addu   %[p_filter_coeffs_p],  %[p_filter_coeffs_p],   %[prec] \n\t"

                "addu   %[p_filter_coeffs_m],  %[p_filter_coeffs_m],   %[prec] \n\t"

                "madd.s %[v],%[v],%[in_val_m], %[fc_val_m]                     \n\t"



                : [v] "=&f" (v),[p_in_p] "+r" (p_in_p), [p_in_m] "+r" (p_in_m),

                  [p_filter_coeffs_p] "+r" (p_filter_coeffs_p),

                  [in_val_p] "=&f" (in_val_p), [in_val_m] "=&f" (in_val_m),

                  [fc_val_p] "=&f" (fc_val_p), [fc_val_m] "=&f" (fc_val_m),

                  [p_filter_coeffs_m] "+r" (p_filter_coeffs_m)

                : [prec] "r" (prec)

                : "memory"

            );

        }

        out[n] = v;

    }

}
