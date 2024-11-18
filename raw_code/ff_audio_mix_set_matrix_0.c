int ff_audio_mix_set_matrix(AudioMix *am, const double *matrix, int stride)

{

    int i, o;



    if ( am->in_channels <= 0 ||  am->in_channels > AVRESAMPLE_MAX_CHANNELS ||

        am->out_channels <= 0 || am->out_channels > AVRESAMPLE_MAX_CHANNELS) {

        av_log(am, AV_LOG_ERROR, "Invalid channel counts\n");

        return AVERROR(EINVAL);

    }



    if (am->matrix) {

        av_free(am->matrix[0]);

        am->matrix = NULL;

    }



#define CONVERT_MATRIX(type, expr)                                          \

    am->matrix_## type[0] = av_mallocz(am->out_channels * am->in_channels * \

                                       sizeof(*am->matrix_## type[0]));     \

    if (!am->matrix_## type[0])                                             \

        return AVERROR(ENOMEM);                                             \

    for (o = 0; o < am->out_channels; o++) {                                \

        if (o > 0)                                                          \

            am->matrix_## type[o] = am->matrix_## type[o - 1] +             \

                                    am->in_channels;                        \

        for (i = 0; i < am->in_channels; i++) {                             \

            double v = matrix[o * stride + i];                              \

            am->matrix_## type[o][i] = expr;                                \

        }                                                                   \

    }                                                                       \

    am->matrix = (void **)am->matrix_## type;



    switch (am->coeff_type) {

    case AV_MIX_COEFF_TYPE_Q8:

        CONVERT_MATRIX(q8, av_clip_int16(lrint(256.0 * v)))

        break;

    case AV_MIX_COEFF_TYPE_Q15:

        CONVERT_MATRIX(q15, av_clipl_int32(llrint(32768.0 * v)))

        break;

    case AV_MIX_COEFF_TYPE_FLT:

        CONVERT_MATRIX(flt, v)

        break;

    default:

        av_log(am, AV_LOG_ERROR, "Invalid mix coeff type\n");

        return AVERROR(EINVAL);

    }



    /* TODO: detect situations where we can just swap around pointers

             instead of doing matrix multiplications with 0.0 and 1.0 */



    return 0;

}
