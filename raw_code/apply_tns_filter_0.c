static void apply_tns_filter(float *out, float *in, int order, int direction,

                             float *tns_coefs, int ltp_used, int w, int filt,

                             int start_i, int len)

{

    int i, j, inc, start = start_i;

    float tmp[TNS_MAX_ORDER+1];

    if (direction) {

        inc = -1;

        start = (start + len) - 1;

    } else {

        inc = 1;

    }

    if (!ltp_used) {    /* AR filter */

        for (i = 0; i < len; i++, start += inc)

            out[i] = in[start];

            for (j = 1; j <= FFMIN(i, order); j++)

                out[i] += tns_coefs[j]*in[start - j*inc];

    } else {            /* MA filter */

        for (i = 0; i < len; i++, start += inc) {

            tmp[0] = out[i] = in[start];

            for (j = 1; j <= FFMIN(i, order); j++)

                out[i] += tmp[j]*tns_coefs[j];

            for (j = order; j > 0; j--)

                tmp[j] = tmp[j - 1];

        }

    }

}
