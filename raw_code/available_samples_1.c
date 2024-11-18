static inline int available_samples(AVFrame *out)

{

    int bytes_per_sample = av_get_bytes_per_sample(out->format);

    int samples = out->linesize[0] / bytes_per_sample;



    if (av_sample_fmt_is_planar(out->format)) {

        return samples;

    } else {

        int channels = av_get_channel_layout_nb_channels(out->channel_layout);

        return samples / channels;

    }

}
