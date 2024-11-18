int ff_audio_mix(AudioMix *am, AudioData *src)

{

    int use_generic = 1;

    int len = src->nb_samples;

    int i, j;



    /* determine whether to use the optimized function based on pointer and

       samples alignment in both the input and output */

    if (am->has_optimized_func) {

        int aligned_len = FFALIGN(len, am->samples_align);

        if (!(src->ptr_align % am->ptr_align) &&

            src->samples_align >= aligned_len) {

            len = aligned_len;

            use_generic = 0;

        }

    }

    av_dlog(am->avr, "audio_mix: %d samples - %d to %d channels (%s)\n",

            src->nb_samples, am->in_channels, am->out_channels,

            use_generic ? am->func_descr_generic : am->func_descr);



    if (am->in_matrix_channels && am->out_matrix_channels) {

        uint8_t **data;

        uint8_t *data0[AVRESAMPLE_MAX_CHANNELS];



        if (am->out_matrix_channels < am->out_channels ||

             am->in_matrix_channels <  am->in_channels) {

            for (i = 0, j = 0; i < FFMAX(am->in_channels, am->out_channels); i++) {

                if (am->input_skip[i] || am->output_skip[i] || am->output_zero[i])

                    continue;

                data0[j++] = src->data[i];

            }

            data = data0;

        } else {

            data = src->data;

        }



        if (use_generic)

            am->mix_generic(data, am->matrix, len, am->out_matrix_channels,

                            am->in_matrix_channels);

        else

            am->mix(data, am->matrix, len, am->out_matrix_channels,

                    am->in_matrix_channels);

    }



    if (am->out_matrix_channels < am->out_channels) {

        for (i = 0; i < am->out_channels; i++)

            if (am->output_zero[i])

                av_samples_set_silence(&src->data[i], 0, len, 1, am->fmt);

    }



    ff_audio_data_set_channels(src, am->out_channels);



    return 0;

}
