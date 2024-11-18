static void predictor_decompress_fir_adapt(int32_t *error_buffer,

                                           int32_t *buffer_out,

                                           int output_size,

                                           int readsamplesize,

                                           int16_t *predictor_coef_table,

                                           int predictor_coef_num,

                                           int predictor_quantitization)

{

    int i;



    /* first sample always copies */

    *buffer_out = *error_buffer;



    if (!predictor_coef_num) {

        if (output_size <= 1)

            return;



        memcpy(&buffer_out[1], &error_buffer[1],

               (output_size - 1) * sizeof(*buffer_out));

        return;

    }



    if (predictor_coef_num == 31) {

        /* simple 1st-order prediction */

        if (output_size <= 1)

            return;

        for (i = 1; i < output_size; i++) {

            buffer_out[i] = sign_extend(buffer_out[i - 1] + error_buffer[i],

                                        readsamplesize);

        }

        return;

    }



    /* read warm-up samples */

    for (i = 0; i < predictor_coef_num; i++) {

        buffer_out[i + 1] = sign_extend(buffer_out[i] + error_buffer[i + 1],

                                        readsamplesize);

    }



    /* NOTE: 4 and 8 are very common cases that could be optimized. */



    /* general case */

    for (i = predictor_coef_num; i < output_size - 1; i++) {

        int j;

        int val = 0;

        int error_val = error_buffer[i + 1];

        int error_sign;

        int d = buffer_out[i - predictor_coef_num];



        for (j = 0; j < predictor_coef_num; j++) {

            val += (buffer_out[i - j] - d) *

                   predictor_coef_table[j];

        }



        val = (val + (1 << (predictor_quantitization - 1))) >>

              predictor_quantitization;

        val += d + error_val;



        buffer_out[i + 1] = sign_extend(val, readsamplesize);



        /* adapt LPC coefficients */

        error_sign = sign_only(error_val);

        if (error_sign) {

            for (j = predictor_coef_num - 1; j >= 0 && error_val * error_sign > 0; j--) {

                int sign;

                val  = d - buffer_out[i - j];

                sign = sign_only(val) * error_sign;

                predictor_coef_table[j] -= sign;

                val *= sign;

                error_val -= ((val >> predictor_quantitization) *

                              (predictor_coef_num - j));

            }

        }

    }

}
