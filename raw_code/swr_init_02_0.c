av_cold int swr_init(struct SwrContext *s){

    int ret;



    clear_context(s);



    if(s-> in_sample_fmt >= AV_SAMPLE_FMT_NB){

        av_log(s, AV_LOG_ERROR, "Requested input sample format %d is invalid\n", s->in_sample_fmt);

        return AVERROR(EINVAL);

    }

    if(s->out_sample_fmt >= AV_SAMPLE_FMT_NB){

        av_log(s, AV_LOG_ERROR, "Requested output sample format %d is invalid\n", s->out_sample_fmt);

        return AVERROR(EINVAL);

    }



    s->out.ch_count  = s-> user_out_ch_count;

    s-> in.ch_count  = s->  user_in_ch_count;

    s->used_ch_count = s->user_used_ch_count;



    s-> in_ch_layout = s-> user_in_ch_layout;

    s->out_ch_layout = s->user_out_ch_layout;



    if(av_get_channel_layout_nb_channels(s-> in_ch_layout) > SWR_CH_MAX) {

        av_log(s, AV_LOG_WARNING, "Input channel layout 0x%"PRIx64" is invalid or unsupported.\n", s-> in_ch_layout);

        s->in_ch_layout = 0;

    }



    if(av_get_channel_layout_nb_channels(s->out_ch_layout) > SWR_CH_MAX) {

        av_log(s, AV_LOG_WARNING, "Output channel layout 0x%"PRIx64" is invalid or unsupported.\n", s->out_ch_layout);

        s->out_ch_layout = 0;

    }



    switch(s->engine){

#if CONFIG_LIBSOXR

        case SWR_ENGINE_SOXR: s->resampler = &swri_soxr_resampler; break;

#endif

        case SWR_ENGINE_SWR : s->resampler = &swri_resampler; break;

        default:

            av_log(s, AV_LOG_ERROR, "Requested resampling engine is unavailable\n");

            return AVERROR(EINVAL);

    }



    if(!s->used_ch_count)

        s->used_ch_count= s->in.ch_count;



    if(s->used_ch_count && s-> in_ch_layout && s->used_ch_count != av_get_channel_layout_nb_channels(s-> in_ch_layout)){

        av_log(s, AV_LOG_WARNING, "Input channel layout has a different number of channels than the number of used channels, ignoring layout\n");

        s-> in_ch_layout= 0;

    }



    if(!s-> in_ch_layout)

        s-> in_ch_layout= av_get_default_channel_layout(s->used_ch_count);

    if(!s->out_ch_layout)

        s->out_ch_layout= av_get_default_channel_layout(s->out.ch_count);



    s->rematrix= s->out_ch_layout  !=s->in_ch_layout || s->rematrix_volume!=1.0 ||

                 s->rematrix_custom;



    if(s->int_sample_fmt == AV_SAMPLE_FMT_NONE){

        if(av_get_planar_sample_fmt(s->in_sample_fmt) <= AV_SAMPLE_FMT_S16P){

            s->int_sample_fmt= AV_SAMPLE_FMT_S16P;

        }else if(   av_get_planar_sample_fmt(s-> in_sample_fmt) == AV_SAMPLE_FMT_S32P

                 && av_get_planar_sample_fmt(s->out_sample_fmt) == AV_SAMPLE_FMT_S32P

                 && !s->rematrix

                 && s->engine != SWR_ENGINE_SOXR){

            s->int_sample_fmt= AV_SAMPLE_FMT_S32P;

        }else if(av_get_planar_sample_fmt(s->in_sample_fmt) <= AV_SAMPLE_FMT_FLTP){

            s->int_sample_fmt= AV_SAMPLE_FMT_FLTP;

        }else{

            av_log(s, AV_LOG_DEBUG, "Using double precision mode\n");

            s->int_sample_fmt= AV_SAMPLE_FMT_DBLP;

        }

    }



    if(   s->int_sample_fmt != AV_SAMPLE_FMT_S16P

        &&s->int_sample_fmt != AV_SAMPLE_FMT_S32P

        &&s->int_sample_fmt != AV_SAMPLE_FMT_FLTP

        &&s->int_sample_fmt != AV_SAMPLE_FMT_DBLP){

        av_log(s, AV_LOG_ERROR, "Requested sample format %s is not supported internally, S16/S32/FLT/DBL is supported\n", av_get_sample_fmt_name(s->int_sample_fmt));

        return AVERROR(EINVAL);

    }



    set_audiodata_fmt(&s-> in, s-> in_sample_fmt);

    set_audiodata_fmt(&s->out, s->out_sample_fmt);



    if (s->firstpts_in_samples != AV_NOPTS_VALUE) {

        if (!s->async && s->min_compensation >= FLT_MAX/2)

            s->async = 1;

        s->firstpts =

        s->outpts   = s->firstpts_in_samples * s->out_sample_rate;

    } else

        s->firstpts = AV_NOPTS_VALUE;



    if (s->async) {

        if (s->min_compensation >= FLT_MAX/2)

            s->min_compensation = 0.001;

        if (s->async > 1.0001) {

            s->max_soft_compensation = s->async / (double) s->in_sample_rate;

        }

    }



    if (s->out_sample_rate!=s->in_sample_rate || (s->flags & SWR_FLAG_RESAMPLE)){

        s->resample = s->resampler->init(s->resample, s->out_sample_rate, s->in_sample_rate, s->filter_size, s->phase_shift, s->linear_interp, s->cutoff, s->int_sample_fmt, s->filter_type, s->kaiser_beta, s->precision, s->cheby);

    }else

        s->resampler->free(&s->resample);

    if(    s->int_sample_fmt != AV_SAMPLE_FMT_S16P

        && s->int_sample_fmt != AV_SAMPLE_FMT_S32P

        && s->int_sample_fmt != AV_SAMPLE_FMT_FLTP

        && s->int_sample_fmt != AV_SAMPLE_FMT_DBLP

        && s->resample){

        av_log(s, AV_LOG_ERROR, "Resampling only supported with internal s16/s32/flt/dbl\n");

        return -1;

    }



#define RSC 1 //FIXME finetune

    if(!s-> in.ch_count)

        s-> in.ch_count= av_get_channel_layout_nb_channels(s-> in_ch_layout);

    if(!s->used_ch_count)

        s->used_ch_count= s->in.ch_count;

    if(!s->out.ch_count)

        s->out.ch_count= av_get_channel_layout_nb_channels(s->out_ch_layout);



    if(!s-> in.ch_count){

        av_assert0(!s->in_ch_layout);

        av_log(s, AV_LOG_ERROR, "Input channel count and layout are unset\n");

        return -1;

    }



    if ((!s->out_ch_layout || !s->in_ch_layout) && s->used_ch_count != s->out.ch_count && !s->rematrix_custom) {

        char l1[1024], l2[1024];

        av_get_channel_layout_string(l1, sizeof(l1), s-> in.ch_count, s-> in_ch_layout);

        av_get_channel_layout_string(l2, sizeof(l2), s->out.ch_count, s->out_ch_layout);

        av_log(s, AV_LOG_ERROR, "Rematrix is needed between %s and %s "

               "but there is not enough information to do it\n", l1, l2);

        return -1;

    }



av_assert0(s->used_ch_count);

av_assert0(s->out.ch_count);

    s->resample_first= RSC*s->out.ch_count/s->in.ch_count - RSC < s->out_sample_rate/(float)s-> in_sample_rate - 1.0;



    s->in_buffer= s->in;

    s->silence  = s->in;

    s->drop_temp= s->out;



    if(!s->resample && !s->rematrix && !s->channel_map && !s->dither.method){

        s->full_convert = swri_audio_convert_alloc(s->out_sample_fmt,

                                                   s-> in_sample_fmt, s-> in.ch_count, NULL, 0);

        return 0;

    }



    s->in_convert = swri_audio_convert_alloc(s->int_sample_fmt,

                                             s-> in_sample_fmt, s->used_ch_count, s->channel_map, 0);

    s->out_convert= swri_audio_convert_alloc(s->out_sample_fmt,

                                             s->int_sample_fmt, s->out.ch_count, NULL, 0);



    if (!s->in_convert || !s->out_convert)

        return AVERROR(ENOMEM);



    s->postin= s->in;

    s->preout= s->out;

    s->midbuf= s->in;



    if(s->channel_map){

        s->postin.ch_count=

        s->midbuf.ch_count= s->used_ch_count;

        if(s->resample)

            s->in_buffer.ch_count= s->used_ch_count;

    }

    if(!s->resample_first){

        s->midbuf.ch_count= s->out.ch_count;

        if(s->resample)

            s->in_buffer.ch_count = s->out.ch_count;

    }



    set_audiodata_fmt(&s->postin, s->int_sample_fmt);

    set_audiodata_fmt(&s->midbuf, s->int_sample_fmt);

    set_audiodata_fmt(&s->preout, s->int_sample_fmt);



    if(s->resample){

        set_audiodata_fmt(&s->in_buffer, s->int_sample_fmt);

    }



    if ((ret = swri_dither_init(s, s->out_sample_fmt, s->int_sample_fmt)) < 0)

        return ret;



    if(s->rematrix || s->dither.method)

        return swri_rematrix_init(s);



    return 0;

}
