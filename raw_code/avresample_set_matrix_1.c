int avresample_set_matrix(AVAudioResampleContext *avr, const double *matrix,

                          int stride)

{

    int in_channels, out_channels, i, o;



    in_channels  = av_get_channel_layout_nb_channels(avr->in_channel_layout);

    out_channels = av_get_channel_layout_nb_channels(avr->out_channel_layout);



    if ( in_channels < 0 ||  in_channels > AVRESAMPLE_MAX_CHANNELS ||

        out_channels < 0 || out_channels > AVRESAMPLE_MAX_CHANNELS) {

        av_log(avr, AV_LOG_ERROR, "Invalid channel layouts\n");

        return AVERROR(EINVAL);

    }



    if (avr->am->matrix)

        av_freep(avr->am->matrix);



#define CONVERT_MATRIX(type, expr)                                          \

    avr->am->matrix_## type[0] = av_mallocz(out_channels * in_channels *    \

                                            sizeof(*avr->am->matrix_## type[0])); \

    if (!avr->am->matrix_## type[0])                                        \

        return AVERROR(ENOMEM);                                             \

    for (o = 0; o < out_channels; o++) {                                    \

        if (o > 0)                                                          \

            avr->am->matrix_## type[o] = avr->am->matrix_## type[o - 1] +   \

                                         in_channels;                       \

        for (i = 0; i < in_channels; i++) {                                 \

            double v = matrix[o * stride + i];                              \

            avr->am->matrix_## type[o][i] = expr;                           \

        }                                                                   \

    }                                                                       \

    avr->am->matrix = (void **)avr->am->matrix_## type;



    switch (avr->mix_coeff_type) {

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

        av_log(avr, AV_LOG_ERROR, "Invalid mix coeff type\n");

        return AVERROR(EINVAL);

    }



    /* TODO: detect situations where we can just swap around pointers

             instead of doing matrix multiplications with 0.0 and 1.0 */



    return 0;

}
