static void sbr_hf_assemble(float Y[2][38][64][2], const float X_high[64][40][2],

                            SpectralBandReplication *sbr, SBRData *ch_data,

                            const int e_a[2])

{

    int e, i, j, m;

    const int h_SL = 4 * !sbr->bs_smoothing_mode;

    const int kx = sbr->kx[1];

    const int m_max = sbr->m[1];

    static const float h_smooth[5] = {

        0.33333333333333,

        0.30150283239582,

        0.21816949906249,

        0.11516383427084,

        0.03183050093751,

    };

    static const int8_t phi[2][4] = {

        {  1,  0, -1,  0}, // real

        {  0,  1,  0, -1}, // imaginary

    };

    float (*g_temp)[48] = ch_data->g_temp, (*q_temp)[48] = ch_data->q_temp;

    int indexnoise = ch_data->f_indexnoise;

    int indexsine  = ch_data->f_indexsine;

    memcpy(Y[0], Y[1], sizeof(Y[0]));



    if (sbr->reset) {

        for (i = 0; i < h_SL; i++) {

            memcpy(g_temp[i + 2*ch_data->t_env[0]], sbr->gain[0], m_max * sizeof(sbr->gain[0][0]));

            memcpy(q_temp[i + 2*ch_data->t_env[0]], sbr->q_m[0],  m_max * sizeof(sbr->q_m[0][0]));

        }

    } else if (h_SL) {

        memcpy(g_temp[2*ch_data->t_env[0]], g_temp[2*ch_data->t_env_num_env_old], 4*sizeof(g_temp[0]));

        memcpy(q_temp[2*ch_data->t_env[0]], q_temp[2*ch_data->t_env_num_env_old], 4*sizeof(q_temp[0]));

    }



    for (e = 0; e < ch_data->bs_num_env; e++) {

        for (i = 2 * ch_data->t_env[e]; i < 2 * ch_data->t_env[e + 1]; i++) {

            memcpy(g_temp[h_SL + i], sbr->gain[e], m_max * sizeof(sbr->gain[0][0]));

            memcpy(q_temp[h_SL + i], sbr->q_m[e],  m_max * sizeof(sbr->q_m[0][0]));

        }

    }



    for (e = 0; e < ch_data->bs_num_env; e++) {

        for (i = 2 * ch_data->t_env[e]; i < 2 * ch_data->t_env[e + 1]; i++) {

            int phi_sign = (1 - 2*(kx & 1));



            if (h_SL && e != e_a[0] && e != e_a[1]) {

                for (m = 0; m < m_max; m++) {

                    const int idx1 = i + h_SL;

                    float g_filt = 0.0f;

                    for (j = 0; j <= h_SL; j++)

                        g_filt += g_temp[idx1 - j][m] * h_smooth[j];

                    Y[1][i][m + kx][0] =

                        X_high[m + kx][i + ENVELOPE_ADJUSTMENT_OFFSET][0] * g_filt;

                    Y[1][i][m + kx][1] =

                        X_high[m + kx][i + ENVELOPE_ADJUSTMENT_OFFSET][1] * g_filt;

                }

            } else {

                for (m = 0; m < m_max; m++) {

                    const float g_filt = g_temp[i + h_SL][m];

                    Y[1][i][m + kx][0] =

                        X_high[m + kx][i + ENVELOPE_ADJUSTMENT_OFFSET][0] * g_filt;

                    Y[1][i][m + kx][1] =

                        X_high[m + kx][i + ENVELOPE_ADJUSTMENT_OFFSET][1] * g_filt;

                }

            }



            if (e != e_a[0] && e != e_a[1]) {

                for (m = 0; m < m_max; m++) {

                    indexnoise = (indexnoise + 1) & 0x1ff;

                    if (sbr->s_m[e][m]) {

                        Y[1][i][m + kx][0] +=

                            sbr->s_m[e][m] * phi[0][indexsine];

                        Y[1][i][m + kx][1] +=

                            sbr->s_m[e][m] * (phi[1][indexsine] * phi_sign);

                    } else {

                        float q_filt;

                        if (h_SL) {

                            const int idx1 = i + h_SL;

                            q_filt = 0.0f;

                            for (j = 0; j <= h_SL; j++)

                                q_filt += q_temp[idx1 - j][m] * h_smooth[j];

                        } else {

                            q_filt = q_temp[i][m];

                        }

                        Y[1][i][m + kx][0] +=

                            q_filt * sbr_noise_table[indexnoise][0];

                        Y[1][i][m + kx][1] +=

                            q_filt * sbr_noise_table[indexnoise][1];

                    }

                    phi_sign = -phi_sign;

                }

            } else {

                indexnoise = (indexnoise + m_max) & 0x1ff;

                for (m = 0; m < m_max; m++) {

                    Y[1][i][m + kx][0] +=

                        sbr->s_m[e][m] * phi[0][indexsine];

                    Y[1][i][m + kx][1] +=

                        sbr->s_m[e][m] * (phi[1][indexsine] * phi_sign);

                    phi_sign = -phi_sign;

                }

            }

            indexsine = (indexsine + 1) & 3;

        }

    }

    ch_data->f_indexnoise = indexnoise;

    ch_data->f_indexsine  = indexsine;

}
