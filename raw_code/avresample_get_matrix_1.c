int avresample_get_matrix(AVAudioResampleContext *avr, double *matrix,

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



    switch (avr->mix_coeff_type) {

    case AV_MIX_COEFF_TYPE_Q8:

        if (!avr->am->matrix_q8[0]) {

            av_log(avr, AV_LOG_ERROR, "matrix is not set\n");

            return AVERROR(EINVAL);

        }

        for (o = 0; o < out_channels; o++)

            for (i = 0; i < in_channels; i++)

                matrix[o * stride + i] = avr->am->matrix_q8[o][i] / 256.0;

        break;

    case AV_MIX_COEFF_TYPE_Q15:

        if (!avr->am->matrix_q15[0]) {

            av_log(avr, AV_LOG_ERROR, "matrix is not set\n");

            return AVERROR(EINVAL);

        }

        for (o = 0; o < out_channels; o++)

            for (i = 0; i < in_channels; i++)

                matrix[o * stride + i] = avr->am->matrix_q15[o][i] / 32768.0;

        break;

    case AV_MIX_COEFF_TYPE_FLT:

        if (!avr->am->matrix_flt[0]) {

            av_log(avr, AV_LOG_ERROR, "matrix is not set\n");

            return AVERROR(EINVAL);

        }

        for (o = 0; o < out_channels; o++)

            for (i = 0; i < in_channels; i++)

                matrix[o * stride + i] = avr->am->matrix_flt[o][i];

        break;

    default:

        av_log(avr, AV_LOG_ERROR, "Invalid mix coeff type\n");

        return AVERROR(EINVAL);

    }

    return 0;

}
