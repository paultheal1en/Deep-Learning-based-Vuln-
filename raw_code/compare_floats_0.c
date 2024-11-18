static int compare_floats(const float *a, const float *b, int len,

                          float max_diff)

{

    int i;

    for (i = 0; i < len; i++) {

        if (fabsf(a[i] - b[i]) > max_diff) {

            av_log(NULL, AV_LOG_ERROR, "%d: %- .12f - %- .12f = % .12g\n",

                   i, a[i], b[i], a[i] - b[i]);

            return -1;

        }

    }

    return 0;

}
