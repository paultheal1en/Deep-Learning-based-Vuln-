static int init_resampler(AVCodecContext *input_codec_context,

                          AVCodecContext *output_codec_context,

                          SwrContext **resample_context)

{

    /**

     * Only initialize the resampler if it is necessary, i.e.,

     * if and only if the sample formats differ.

     */

    if (input_codec_context->sample_fmt != output_codec_context->sample_fmt ||

        input_codec_context->channels != output_codec_context->channels) {

        int error;



        /**

         * Create a resampler context for the conversion.

         * Set the conversion parameters.

         * Default channel layouts based on the number of channels

         * are assumed for simplicity (they are sometimes not detected

         * properly by the demuxer and/or decoder).

         */

        *resample_context = swr_alloc_set_opts(NULL,

                                              av_get_default_channel_layout(output_codec_context->channels),

                                              output_codec_context->sample_fmt,

                                              output_codec_context->sample_rate,

                                              av_get_default_channel_layout(input_codec_context->channels),

                                              input_codec_context->sample_fmt,

                                              input_codec_context->sample_rate,

                                              0, NULL);

        if (!*resample_context) {

            fprintf(stderr, "Could not allocate resample context\n");

            return AVERROR(ENOMEM);

        }

        /**

        * Perform a sanity check so that the number of converted samples is

        * not greater than the number of samples to be converted.

        * If the sample rates differ, this case has to be handled differently

        */

        av_assert0(output_codec_context->sample_rate == input_codec_context->sample_rate);



        /** Open the resampler with the specified parameters. */

        if ((error = swr_init(*resample_context)) < 0) {

            fprintf(stderr, "Could not open resample context\n");

            swr_free(resample_context);

            return error;

        }

    }

    return 0;

}
