static void get_fcb_param(FCBParam *optim, int16_t *impulse_resp,

                          int16_t *buf, int pulse_cnt, int pitch_lag)

{

    FCBParam param;

    int16_t impulse_r[SUBFRAME_LEN];

    int16_t temp_corr[SUBFRAME_LEN];

    int16_t impulse_corr[SUBFRAME_LEN];



    int ccr1[SUBFRAME_LEN];

    int ccr2[SUBFRAME_LEN];

    int amp, err, max, max_amp_index, min, scale, i, j, k, l;



    int64_t temp;



    /* Update impulse response */

    memcpy(impulse_r, impulse_resp, sizeof(int16_t) * SUBFRAME_LEN);

    param.dirac_train = 0;

    if (pitch_lag < SUBFRAME_LEN - 2) {

        param.dirac_train = 1;

        gen_dirac_train(impulse_r, pitch_lag);

    }



    for (i = 0; i < SUBFRAME_LEN; i++)

        temp_corr[i] = impulse_r[i] >> 1;



    /* Compute impulse response autocorrelation */

    temp = dot_product(temp_corr, temp_corr, SUBFRAME_LEN);



    scale = normalize_bits_int32(temp);

    impulse_corr[0] = av_clipl_int32((temp << scale) + (1 << 15)) >> 16;



    for (i = 1; i < SUBFRAME_LEN; i++) {

        temp = dot_product(temp_corr + i, temp_corr, SUBFRAME_LEN - i);

        impulse_corr[i] = av_clipl_int32((temp << scale) + (1 << 15)) >> 16;

    }



    /* Compute crosscorrelation of impulse response with residual signal */

    scale -= 4;

    for (i = 0; i < SUBFRAME_LEN; i++){

        temp = dot_product(buf + i, impulse_r, SUBFRAME_LEN - i);

        if (scale < 0)

            ccr1[i] = temp >> -scale;

        else

            ccr1[i] = av_clipl_int32(temp << scale);

    }



    /* Search loop */

    for (i = 0; i < GRID_SIZE; i++) {

        /* Maximize the crosscorrelation */

        max = 0;

        for (j = i; j < SUBFRAME_LEN; j += GRID_SIZE) {

            temp = FFABS(ccr1[j]);

            if (temp >= max) {

                max = temp;

                param.pulse_pos[0] = j;

            }

        }



        /* Quantize the gain (max crosscorrelation/impulse_corr[0]) */

        amp = max;

        min = 1 << 30;

        max_amp_index = GAIN_LEVELS - 2;

        for (j = max_amp_index; j >= 2; j--) {

            temp = av_clipl_int32((int64_t)fixed_cb_gain[j] *

                                  impulse_corr[0] << 1);

            temp = FFABS(temp - amp);

            if (temp < min) {

                min = temp;

                max_amp_index = j;

            }

        }



        max_amp_index--;

        /* Select additional gain values */

        for (j = 1; j < 5; j++) {

            for (k = i; k < SUBFRAME_LEN; k += GRID_SIZE) {

                temp_corr[k] = 0;

                ccr2[k]      = ccr1[k];

            }

            param.amp_index = max_amp_index + j - 2;

            amp = fixed_cb_gain[param.amp_index];



            param.pulse_sign[0] = (ccr2[param.pulse_pos[0]] < 0) ? -amp : amp;

            temp_corr[param.pulse_pos[0]] = 1;



            for (k = 1; k < pulse_cnt; k++) {

                max = -1 << 30;

                for (l = i; l < SUBFRAME_LEN; l += GRID_SIZE) {

                    if (temp_corr[l])

                        continue;

                    temp = impulse_corr[FFABS(l - param.pulse_pos[k - 1])];

                    temp = av_clipl_int32((int64_t)temp *

                                          param.pulse_sign[k - 1] << 1);

                    ccr2[l] -= temp;

                    temp = FFABS(ccr2[l]);

                    if (temp > max) {

                        max = temp;

                        param.pulse_pos[k] = l;

                    }

                }



                param.pulse_sign[k] = (ccr2[param.pulse_pos[k]] < 0) ?

                                      -amp : amp;

                temp_corr[param.pulse_pos[k]] = 1;

            }



            /* Create the error vector */

            memset(temp_corr, 0, sizeof(int16_t) * SUBFRAME_LEN);



            for (k = 0; k < pulse_cnt; k++)

                temp_corr[param.pulse_pos[k]] = param.pulse_sign[k];



            for (k = SUBFRAME_LEN - 1; k >= 0; k--) {

                temp = 0;

                for (l = 0; l <= k; l++) {

                    int prod = av_clipl_int32((int64_t)temp_corr[l] *

                                              impulse_r[k - l] << 1);

                    temp     = av_clipl_int32(temp + prod);

                }

                temp_corr[k] = temp << 2 >> 16;

            }



            /* Compute square of error */

            err = 0;

            for (k = 0; k < SUBFRAME_LEN; k++) {

                int64_t prod;

                prod = av_clipl_int32((int64_t)buf[k] * temp_corr[k] << 1);

                err  = av_clipl_int32(err - prod);

                prod = av_clipl_int32((int64_t)temp_corr[k] * temp_corr[k]);

                err  = av_clipl_int32(err + prod);

            }



            /* Minimize */

            if (err < optim->min_err) {

                optim->min_err     = err;

                optim->grid_index  = i;

                optim->amp_index   = param.amp_index;

                optim->dirac_train = param.dirac_train;



                for (k = 0; k < pulse_cnt; k++) {

                    optim->pulse_sign[k] = param.pulse_sign[k];

                    optim->pulse_pos[k]  = param.pulse_pos[k];

                }

            }

        }

    }

}
