static void FUNC(flac_decorrelate_indep_c)(uint8_t **out, int32_t **in,

                                           int channels, int len, int shift)

{

    sample *samples = (sample *) OUT(out);

    int i, j;



    for (j = 0; j < len; j++)

        for (i = 0; i < channels; i++)

            S(samples, i, j) = in[i][j] << shift;

}
