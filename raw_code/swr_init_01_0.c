int swr_init(SwrContext *s){

    s->in_buffer_index= 0;

    s->in_buffer_count= 0;

    s->resample_in_constraint= 0;

    free_temp(&s->postin);

    free_temp(&s->midbuf);

    free_temp(&s->preout);

    free_temp(&s->in_buffer);

    swr_audio_convert_free(&s-> in_convert);

    swr_audio_convert_free(&s->out_convert);



    s-> in.planar= s-> in_sample_fmt >= 0x100;

    s->out.planar= s->out_sample_fmt >= 0x100;

    s-> in_sample_fmt &= 0xFF;

    s->out_sample_fmt &= 0xFF;



    //We assume AVOptions checked the various values and the defaults where allowed

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

        s->resample = swr_resample_init(s->resample, s->out_sample_rate, s->in_sample_rate, 16, 10, 0, 0.8);

    }else

        swr_resample_free(&s->resample);

    if(s->int_sample_fmt != AV_SAMPLE_FMT_S16 && s->resample){

        av_log(s, AV_LOG_ERROR, "Resampling only supported with internal s16 currently\n"); //FIXME

        return -1;

    }



    if(!s-> in_ch_layout)

        s-> in_ch_layout= guess_layout(s->in.ch_count);

    if(!s->out_ch_layout)

        s->out_ch_layout= guess_layout(s->out.ch_count);



    s->rematrix= s->out_ch_layout  !=s->in_ch_layout;



#define RSC 1 //FIXME finetune

    if(!s-> in.ch_count)

        s-> in.ch_count= av_get_channel_layout_nb_channels(s-> in_ch_layout);

    if(!s->out.ch_count)

        s->out.ch_count= av_get_channel_layout_nb_channels(s->out_ch_layout);



av_assert0(s-> in.ch_count);

av_assert0(s->out.ch_count);

    s->resample_first= RSC*s->out.ch_count/s->in.ch_count - RSC < s->out_sample_rate/(float)s-> in_sample_rate - 1.0;



    s-> in.bps= av_get_bits_per_sample_fmt(s-> in_sample_fmt)/8;

    s->int_bps= av_get_bits_per_sample_fmt(s->int_sample_fmt)/8;

    s->out.bps= av_get_bits_per_sample_fmt(s->out_sample_fmt)/8;



    s->in_convert = swr_audio_convert_alloc(s->int_sample_fmt,

                                            s-> in_sample_fmt, s-> in.ch_count, 0);

    s->out_convert= swr_audio_convert_alloc(s->out_sample_fmt,

                                            s->int_sample_fmt, s->out.ch_count, 0);





    s->postin= s->in;

    s->preout= s->out;

    s->midbuf= s->in;

    s->in_buffer= s->in;

    if(!s->resample_first){

        s->midbuf.ch_count= s->out.ch_count;

        s->in_buffer.ch_count = s->out.ch_count;

    }



    s->in_buffer.bps = s->postin.bps = s->midbuf.bps = s->preout.bps =  s->int_bps;

    s->in_buffer.planar = s->postin.planar = s->midbuf.planar = s->preout.planar =  1;





    if(s->rematrix && swr_rematrix_init(s)<0)

        return -1;



    return 0;

}
