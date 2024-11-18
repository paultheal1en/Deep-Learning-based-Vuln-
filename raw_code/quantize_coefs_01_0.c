static inline void quantize_coefs(double *coef, int *idx, float *lpc, int order,

                                  int c_bits)

{

    int i;

    const float *quant_arr = tns_tmp2_map[c_bits];

    for (i = 0; i < order; i++) {

        idx[i] = quant_array_idx((float)coef[i], quant_arr, c_bits ? 16 : 8);

        lpc[i] = quant_arr[idx[i]];

    }

}
