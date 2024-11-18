int av_asrc_buffer_add_buffer(AVFilterContext *ctx,

                              uint8_t *buf, int buf_size, int sample_rate,

                              int sample_fmt, int64_t channel_layout, int planar,

                              int64_t pts, int av_unused flags)

{

    uint8_t *data[8];

    int linesize[8];

    int nb_channels = av_get_channel_layout_nb_channels(channel_layout),

        nb_samples  = buf_size / nb_channels / av_get_bytes_per_sample(sample_fmt);



    av_samples_fill_arrays(data, linesize,

                           buf, nb_channels, nb_samples,

                           sample_fmt, 16);



    return av_asrc_buffer_add_samples(ctx,

                                      data, linesize, nb_samples,

                                      sample_rate,

                                      sample_fmt, channel_layout, planar,

                                      pts, flags);

}
