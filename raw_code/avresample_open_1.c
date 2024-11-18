int avresample_open(AVAudioResampleContext *avr)

{

    int ret;



    /* set channel mixing parameters */

    avr->in_channels = av_get_channel_layout_nb_channels(avr->in_channel_layout);

    if (avr->in_channels <= 0 || avr->in_channels > AVRESAMPLE_MAX_CHANNELS) {

        av_log(avr, AV_LOG_ERROR, "Invalid input channel layout: %"PRIu64"\n",

               avr->in_channel_layout);

        return AVERROR(EINVAL);

    }

    avr->out_channels = av_get_channel_layout_nb_channels(avr->out_channel_layout);

    if (avr->out_channels <= 0 || avr->out_channels > AVRESAMPLE_MAX_CHANNELS) {

        av_log(avr, AV_LOG_ERROR, "Invalid output channel layout: %"PRIu64"\n",

               avr->out_channel_layout);

        return AVERROR(EINVAL);

    }

    avr->resample_channels = FFMIN(avr->in_channels, avr->out_channels);

    avr->downmix_needed    = avr->in_channels  > avr->out_channels;

    avr->upmix_needed      = avr->out_channels > avr->in_channels ||

                             avr->am->matrix                      ||

                             (avr->out_channels == avr->in_channels &&

                              avr->in_channel_layout != avr->out_channel_layout);

    avr->mixing_needed     = avr->downmix_needed || avr->upmix_needed;



    /* set resampling parameters */

    avr->resample_needed   = avr->in_sample_rate != avr->out_sample_rate ||

                             avr->force_resampling;



    /* select internal sample format if not specified by the user */

    if (avr->internal_sample_fmt == AV_SAMPLE_FMT_NONE &&

        (avr->mixing_needed || avr->resample_needed)) {

        enum AVSampleFormat  in_fmt = av_get_planar_sample_fmt(avr->in_sample_fmt);

        enum AVSampleFormat out_fmt = av_get_planar_sample_fmt(avr->out_sample_fmt);

        int max_bps = FFMAX(av_get_bytes_per_sample(in_fmt),

                            av_get_bytes_per_sample(out_fmt));

        if (max_bps <= 2) {

            avr->internal_sample_fmt = AV_SAMPLE_FMT_S16P;

        } else if (avr->mixing_needed) {

            avr->internal_sample_fmt = AV_SAMPLE_FMT_FLTP;

        } else {

            if (max_bps <= 4) {

                if (in_fmt  == AV_SAMPLE_FMT_S32P ||

                    out_fmt == AV_SAMPLE_FMT_S32P) {

                    if (in_fmt  == AV_SAMPLE_FMT_FLTP ||

                        out_fmt == AV_SAMPLE_FMT_FLTP) {

                        /* if one is s32 and the other is flt, use dbl */

                        avr->internal_sample_fmt = AV_SAMPLE_FMT_DBLP;

                    } else {

                        /* if one is s32 and the other is s32, s16, or u8, use s32 */

                        avr->internal_sample_fmt = AV_SAMPLE_FMT_S32P;

                    }

                } else {

                    /* if one is flt and the other is flt, s16 or u8, use flt */

                    avr->internal_sample_fmt = AV_SAMPLE_FMT_FLTP;

                }

            } else {

                /* if either is dbl, use dbl */

                avr->internal_sample_fmt = AV_SAMPLE_FMT_DBLP;

            }

        }

        av_log(avr, AV_LOG_DEBUG, "Using %s as internal sample format\n",

               av_get_sample_fmt_name(avr->internal_sample_fmt));

    }



    /* set sample format conversion parameters */

    if (avr->in_channels == 1)

        avr->in_sample_fmt = av_get_planar_sample_fmt(avr->in_sample_fmt);

    if (avr->out_channels == 1)

        avr->out_sample_fmt = av_get_planar_sample_fmt(avr->out_sample_fmt);

    avr->in_convert_needed = (avr->resample_needed || avr->mixing_needed) &&

                              avr->in_sample_fmt != avr->internal_sample_fmt;

    if (avr->resample_needed || avr->mixing_needed)

        avr->out_convert_needed = avr->internal_sample_fmt != avr->out_sample_fmt;

    else

        avr->out_convert_needed = avr->in_sample_fmt != avr->out_sample_fmt;



    /* allocate buffers */

    if (avr->mixing_needed || avr->in_convert_needed) {

        avr->in_buffer = ff_audio_data_alloc(FFMAX(avr->in_channels, avr->out_channels),

                                             0, avr->internal_sample_fmt,

                                             "in_buffer");

        if (!avr->in_buffer) {

            ret = AVERROR(EINVAL);

            goto error;

        }

    }

    if (avr->resample_needed) {

        avr->resample_out_buffer = ff_audio_data_alloc(avr->out_channels,

                                                       0, avr->internal_sample_fmt,

                                                       "resample_out_buffer");

        if (!avr->resample_out_buffer) {

            ret = AVERROR(EINVAL);

            goto error;

        }

    }

    if (avr->out_convert_needed) {

        avr->out_buffer = ff_audio_data_alloc(avr->out_channels, 0,

                                              avr->out_sample_fmt, "out_buffer");

        if (!avr->out_buffer) {

            ret = AVERROR(EINVAL);

            goto error;

        }

    }

    avr->out_fifo = av_audio_fifo_alloc(avr->out_sample_fmt, avr->out_channels,

                                        1024);

    if (!avr->out_fifo) {

        ret = AVERROR(ENOMEM);

        goto error;

    }



    /* setup contexts */

    if (avr->in_convert_needed) {

        avr->ac_in = ff_audio_convert_alloc(avr, avr->internal_sample_fmt,

                                            avr->in_sample_fmt, avr->in_channels);

        if (!avr->ac_in) {

            ret = AVERROR(ENOMEM);

            goto error;

        }

    }

    if (avr->out_convert_needed) {

        enum AVSampleFormat src_fmt;

        if (avr->in_convert_needed)

            src_fmt = avr->internal_sample_fmt;

        else

            src_fmt = avr->in_sample_fmt;

        avr->ac_out = ff_audio_convert_alloc(avr, avr->out_sample_fmt, src_fmt,

                                             avr->out_channels);

        if (!avr->ac_out) {

            ret = AVERROR(ENOMEM);

            goto error;

        }

    }

    if (avr->resample_needed) {

        avr->resample = ff_audio_resample_init(avr);

        if (!avr->resample) {

            ret = AVERROR(ENOMEM);

            goto error;

        }

    }

    if (avr->mixing_needed) {

        ret = ff_audio_mix_init(avr);

        if (ret < 0)

            goto error;

    }



    return 0;



error:

    avresample_close(avr);

    return ret;

}
