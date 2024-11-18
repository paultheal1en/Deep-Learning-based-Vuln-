int swr_init(struct SwrContext *s){

    s->in_buffer_index= 0;

    s->in_buffer_count= 0;

    s->resample_in_constraint= 0;

    free_temp(&s->postin);

    free_temp(&s->midbuf);

    free_temp(&s->preout);

    free_temp(&s->in_buffer);

    swri_audio_convert_free(&s-> in_convert);

    swri_audio_convert_free(&s->out_convert);

    swri_audio_convert_free(&s->full_convert);



    s-> in.planar= av_sample_fmt_is_planar(s-> in_sample_fmt);

    s->out.planar= av_sample_fmt_is_planar(s->out_sample_fmt);

    s-> in_sample_fmt= av_get_alt_sample_fmt(s-> in_sample_fmt, 0);

    s->out_sample_fmt= av_get_alt_sample_fmt(s->out_sample_fmt, 0);



    if(s-> in_sample_fmt >= AV_SAMPLE_FMT_NB){

        av_log(s, AV_LOG_ERROR, "Requested sample format %s is invalid\n", av_get_sample_fmt_name(s->in_sample_fmt));

        return AVERROR(EINVAL);

    }

    if(s->out_sample_fmt >= AV_SAMPLE_FMT_NB){

        av_log(s, AV_LOG_ERROR, "Requested sample format %s is invalid\n", av_get_sample_fmt_name(s->out_sample_fmt));

        return AVERROR(EINVAL);

    }



    if(   s->int_sample_fmt != AV_SAMPLE_FMT_S16

        &&s->int_sample_fmt != AV_SAMPLE_FMT_FLT){

        av_log(s, AV_LOG_ERROR, "Requested sample format %s is not supported internally, only float & S16 is supported\n", av_get_sample_fmt_name(s->int_sample_fmt));

        return AVERROR(EINVAL);

    }



    //FIXME should we allow/support using FLT on material that doesnt need it ?

    if(s->in_sample_fmt <= AV_SAMPLE_FMT_S16 || s->int_sample_fmt==AV_SAMPLE_FMT_S16){

        s->int_sample_fmt= AV_SAMPLE_FMT_S16;

    }else

        s->int_sample_fmt= AV_SAMPLE_FMT_FLT;





    if (s->out_sample_rate!=s->in_sample_rate || (s->flags & SWR_FLAG_RESAMPLE)){

        s->resample = swri_resample_init(s->resample, s->out_sample_rate, s->in_sample_rate, 16, 10, 0, 0.8);

    }else

        swri_resample_free(&s->resample);

    if(s->int_sample_fmt != AV_SAMPLE_FMT_S16 && s->resample){

        av_log(s, AV_LOG_ERROR, "Resampling only supported with internal s16 currently\n"); //FIXME

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



    s->rematrix= s->out_ch_layout  !=s->in_ch_layout || s->rematrix_volume!=1.0;



#define RSC 1 //FIXME finetune

    if(!s-> in.ch_count)

        s-> in.ch_count= av_get_channel_layout_nb_channels(s-> in_ch_layout);

    if(!s->used_ch_count)

        s->used_ch_count= s->in.ch_count;

    if(!s->out.ch_count)

        s->out.ch_count= av_get_channel_layout_nb_channels(s->out_ch_layout);



av_assert0(s-> in.ch_count);

av_assert0(s->used_ch_count);

av_assert0(s->out.ch_count);

    s->resample_first= RSC*s->out.ch_count/s->in.ch_count - RSC < s->out_sample_rate/(float)s-> in_sample_rate - 1.0;



    s-> in.bps= av_get_bytes_per_sample(s-> in_sample_fmt);

    s->int_bps= av_get_bytes_per_sample(s->int_sample_fmt);

    s->out.bps= av_get_bytes_per_sample(s->out_sample_fmt);



    if(!s->resample && !s->rematrix && !s->channel_map){

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

    s->in_buffer= s->in;

    if(s->channel_map){

        s->postin.ch_count=

        s->midbuf.ch_count=

        s->in_buffer.ch_count= s->used_ch_count;

    }

    if(!s->resample_first){

        s->midbuf.ch_count= s->out.ch_count;

        s->in_buffer.ch_count = s->out.ch_count;

    }



    s->in_buffer.bps = s->postin.bps = s->midbuf.bps = s->preout.bps =  s->int_bps;

    s->in_buffer.planar = s->postin.planar = s->midbuf.planar = s->preout.planar =  1;





    if(s->rematrix)

        return swri_rematrix_init(s);



    return 0;

}
