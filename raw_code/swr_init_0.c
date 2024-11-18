int swr_init(struct SwrContext *s){

    s->in_buffer_index= 0;

    s->in_buffer_count= 0;

    s->resample_in_constraint= 0;

    free_temp(&s->postin);

    free_temp(&s->midbuf);

    free_temp(&s->preout);

    free_temp(&s->in_buffer);

    free_temp(&s->dither);

    swri_audio_convert_free(&s-> in_convert);

    swri_audio_convert_free(&s->out_convert);

    swri_audio_convert_free(&s->full_convert);

    swri_rematrix_free(s);



    s->flushed = 0;



    if(s-> in_sample_fmt >= AV_SAMPLE_FMT_NB){

        av_log(s, AV_LOG_ERROR, "Requested input sample format %d is invalid\n", s->in_sample_fmt);

        return AVERROR(EINVAL);

    }

    if(s->out_sample_fmt >= AV_SAMPLE_FMT_NB){

        av_log(s, AV_LOG_ERROR, "Requested output sample format %d is invalid\n", s->out_sample_fmt);

        return AVERROR(EINVAL);

    }



    //FIXME should we allow/support using FLT on material that doesnt need it ?

    if(av_get_planar_sample_fmt(s->in_sample_fmt) <= AV_SAMPLE_FMT_S16P || s->int_sample_fmt==AV_SAMPLE_FMT_S16P){

        s->int_sample_fmt= AV_SAMPLE_FMT_S16P;

    }else

        s->int_sample_fmt= AV_SAMPLE_FMT_FLTP;



    if(   s->int_sample_fmt != AV_SAMPLE_FMT_S16P

        &&s->int_sample_fmt != AV_SAMPLE_FMT_S32P

        &&s->int_sample_fmt != AV_SAMPLE_FMT_FLTP){

        av_log(s, AV_LOG_ERROR, "Requested sample format %s is not supported internally, S16/S32/FLT is supported\n", av_get_sample_fmt_name(s->int_sample_fmt));

        return AVERROR(EINVAL);

    }



    set_audiodata_fmt(&s-> in, s-> in_sample_fmt);

    set_audiodata_fmt(&s->out, s->out_sample_fmt);



    if (s->out_sample_rate!=s->in_sample_rate || (s->flags & SWR_FLAG_RESAMPLE)){

        s->resample = swri_resample_init(s->resample, s->out_sample_rate, s->in_sample_rate, s->filter_size, s->phase_shift, s->linear_interp, s->cutoff, s->int_sample_fmt);

    }else

        swri_resample_free(&s->resample);

    if(    s->int_sample_fmt != AV_SAMPLE_FMT_S16P

        && s->int_sample_fmt != AV_SAMPLE_FMT_S32P

        && s->int_sample_fmt != AV_SAMPLE_FMT_FLTP

        && s->resample){

        av_log(s, AV_LOG_ERROR, "Resampling only supported with internal s16/s32/flt\n");

        return -1;

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

        av_log(s, AV_LOG_ERROR, "Rematrix is needed but there is not enough information to do it\n");

        return -1;

    }



av_assert0(s->used_ch_count);

av_assert0(s->out.ch_count);

    s->resample_first= RSC*s->out.ch_count/s->in.ch_count - RSC < s->out_sample_rate/(float)s-> in_sample_rate - 1.0;



    s->in_buffer= s->in;



    if(!s->resample && !s->rematrix && !s->channel_map && !s->dither_method){

        s->full_convert = swri_audio_convert_alloc(s->out_sample_fmt,

                                                   s-> in_sample_fmt, s-> in.ch_count, NULL, 0);

        return 0;

    }



    s->in_convert = swri_audio_convert_alloc(s->int_sample_fmt,

                                             s-> in_sample_fmt, s->used_ch_count, s->channel_map, 0);

    s->out_convert= swri_audio_convert_alloc(s->out_sample_fmt,

                                             s->int_sample_fmt, s->out.ch_count, NULL, 0);





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



    s->dither = s->preout;



    if(s->rematrix || s->dither_method)

        return swri_rematrix_init(s);



    return 0;

}
