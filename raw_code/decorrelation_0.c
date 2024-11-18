static void decorrelation(PSContext *ps, float (*out)[32][2], const float (*s)[32][2], int is34)

{

    float power[34][PS_QMF_TIME_SLOTS] = {{0}};

    float transient_gain[34][PS_QMF_TIME_SLOTS];

    float *peak_decay_nrg = ps->peak_decay_nrg;

    float *power_smooth = ps->power_smooth;

    float *peak_decay_diff_smooth = ps->peak_decay_diff_smooth;

    float (*delay)[PS_QMF_TIME_SLOTS + PS_MAX_DELAY][2] = ps->delay;

    float (*ap_delay)[PS_AP_LINKS][PS_QMF_TIME_SLOTS + PS_MAX_AP_DELAY][2] = ps->ap_delay;

    const int8_t *k_to_i = is34 ? k_to_i_34 : k_to_i_20;

    const float peak_decay_factor = 0.76592833836465f;

    const float transient_impact  = 1.5f;

    const float a_smooth          = 0.25f; ///< Smoothing coefficient

    int i, k, m, n;

    int n0 = 0, nL = 32;

    static const int link_delay[] = { 3, 4, 5 };

    static const float a[] = { 0.65143905753106f,

                               0.56471812200776f,

                               0.48954165955695f };



    if (is34 != ps->is34bands_old) {

        memset(ps->peak_decay_nrg,         0, sizeof(ps->peak_decay_nrg));

        memset(ps->power_smooth,           0, sizeof(ps->power_smooth));

        memset(ps->peak_decay_diff_smooth, 0, sizeof(ps->peak_decay_diff_smooth));

        memset(ps->delay,                  0, sizeof(ps->delay));

        memset(ps->ap_delay,               0, sizeof(ps->ap_delay));

    }



    for (n = n0; n < nL; n++) {

        for (k = 0; k < NR_BANDS[is34]; k++) {

            int i = k_to_i[k];

            power[i][n] += s[k][n][0] * s[k][n][0] + s[k][n][1] * s[k][n][1];

        }

    }



    //Transient detection

    for (i = 0; i < NR_PAR_BANDS[is34]; i++) {

        for (n = n0; n < nL; n++) {

            float decayed_peak = peak_decay_factor * peak_decay_nrg[i];

            float denom;

            peak_decay_nrg[i] = FFMAX(decayed_peak, power[i][n]);

            power_smooth[i] += a_smooth * (power[i][n] - power_smooth[i]);

            peak_decay_diff_smooth[i] += a_smooth * (peak_decay_nrg[i] - power[i][n] - peak_decay_diff_smooth[i]);

            denom = transient_impact * peak_decay_diff_smooth[i];

            transient_gain[i][n]   = (denom > power_smooth[i]) ?

                                         power_smooth[i] / denom : 1.0f;

        }

    }



    //Decorrelation and transient reduction

    //                         PS_AP_LINKS - 1

    //                               -----

    //                                | |  Q_fract_allpass[k][m]*z^-link_delay[m] - a[m]*g_decay_slope[k]

    //H[k][z] = z^-2 * phi_fract[k] * | | ----------------------------------------------------------------

    //                                | | 1 - a[m]*g_decay_slope[k]*Q_fract_allpass[k][m]*z^-link_delay[m]

    //                               m = 0

    //d[k][z] (out) = transient_gain_mapped[k][z] * H[k][z] * s[k][z]

    for (k = 0; k < NR_ALLPASS_BANDS[is34]; k++) {

        int b = k_to_i[k];

        float g_decay_slope = 1.f - DECAY_SLOPE * (k - DECAY_CUTOFF[is34]);

        float ag[PS_AP_LINKS];

        g_decay_slope = av_clipf(g_decay_slope, 0.f, 1.f);

        memcpy(delay[k], delay[k]+nL, PS_MAX_DELAY*sizeof(delay[k][0]));

        memcpy(delay[k]+PS_MAX_DELAY, s[k], numQMFSlots*sizeof(delay[k][0]));

        for (m = 0; m < PS_AP_LINKS; m++) {

            memcpy(ap_delay[k][m],   ap_delay[k][m]+numQMFSlots,           5*sizeof(ap_delay[k][m][0]));

            ag[m] = a[m] * g_decay_slope;

        }

        for (n = n0; n < nL; n++) {

            float in_re = delay[k][n+PS_MAX_DELAY-2][0] * phi_fract[is34][k][0] -

                          delay[k][n+PS_MAX_DELAY-2][1] * phi_fract[is34][k][1];

            float in_im = delay[k][n+PS_MAX_DELAY-2][0] * phi_fract[is34][k][1] +

                          delay[k][n+PS_MAX_DELAY-2][1] * phi_fract[is34][k][0];

            for (m = 0; m < PS_AP_LINKS; m++) {

                float a_re                = ag[m] * in_re;

                float a_im                = ag[m] * in_im;

                float link_delay_re       = ap_delay[k][m][n+5-link_delay[m]][0];

                float link_delay_im       = ap_delay[k][m][n+5-link_delay[m]][1];

                float fractional_delay_re = Q_fract_allpass[is34][k][m][0];

                float fractional_delay_im = Q_fract_allpass[is34][k][m][1];

                ap_delay[k][m][n+5][0] = in_re;

                ap_delay[k][m][n+5][1] = in_im;

                in_re = link_delay_re * fractional_delay_re - link_delay_im * fractional_delay_im - a_re;

                in_im = link_delay_re * fractional_delay_im + link_delay_im * fractional_delay_re - a_im;

                ap_delay[k][m][n+5][0] += ag[m] * in_re;

                ap_delay[k][m][n+5][1] += ag[m] * in_im;

            }

            out[k][n][0] = transient_gain[b][n] * in_re;

            out[k][n][1] = transient_gain[b][n] * in_im;

        }

    }

    for (; k < SHORT_DELAY_BAND[is34]; k++) {

        memcpy(delay[k], delay[k]+nL, PS_MAX_DELAY*sizeof(delay[k][0]));

        memcpy(delay[k]+PS_MAX_DELAY, s[k], numQMFSlots*sizeof(delay[k][0]));

        for (n = n0; n < nL; n++) {

            //H = delay 14

            out[k][n][0] = transient_gain[k_to_i[k]][n] * delay[k][n+PS_MAX_DELAY-14][0];

            out[k][n][1] = transient_gain[k_to_i[k]][n] * delay[k][n+PS_MAX_DELAY-14][1];

        }

    }

    for (; k < NR_BANDS[is34]; k++) {

        memcpy(delay[k], delay[k]+nL, PS_MAX_DELAY*sizeof(delay[k][0]));

        memcpy(delay[k]+PS_MAX_DELAY, s[k], numQMFSlots*sizeof(delay[k][0]));

        for (n = n0; n < nL; n++) {

            //H = delay 1

            out[k][n][0] = transient_gain[k_to_i[k]][n] * delay[k][n+PS_MAX_DELAY-1][0];

            out[k][n][1] = transient_gain[k_to_i[k]][n] * delay[k][n+PS_MAX_DELAY-1][1];

        }

    }

}
