static int compute_pkt_fields2(AVFormatContext *s, AVStream *st, AVPacket *pkt){

    int delay = FFMAX(st->codec->has_b_frames, !!st->codec->max_b_frames);

    int num, den, frame_size, i;



    av_dlog(s, "compute_pkt_fields2: pts:%"PRId64" dts:%"PRId64" cur_dts:%"PRId64" b:%d size:%d st:%d\n",

            pkt->pts, pkt->dts, st->cur_dts, delay, pkt->size, pkt->stream_index);



/*    if(pkt->pts == AV_NOPTS_VALUE && pkt->dts == AV_NOPTS_VALUE)

        return AVERROR(EINVAL);*/



    /* duration field */

    if (pkt->duration == 0) {

        compute_frame_duration(&num, &den, st, NULL, pkt);

        if (den && num) {

            pkt->duration = av_rescale(1, num * (int64_t)st->time_base.den * st->codec->ticks_per_frame, den * (int64_t)st->time_base.num);

        }

    }



    if(pkt->pts == AV_NOPTS_VALUE && pkt->dts != AV_NOPTS_VALUE && delay==0)

        pkt->pts= pkt->dts;



    //XXX/FIXME this is a temporary hack until all encoders output pts

    if((pkt->pts == 0 || pkt->pts == AV_NOPTS_VALUE) && pkt->dts == AV_NOPTS_VALUE && !delay){

        pkt->dts=

//        pkt->pts= st->cur_dts;

        pkt->pts= st->pts.val;

    }



    //calculate dts from pts

    if(pkt->pts != AV_NOPTS_VALUE && pkt->dts == AV_NOPTS_VALUE && delay <= MAX_REORDER_DELAY){

        st->pts_buffer[0]= pkt->pts;

        for(i=1; i<delay+1 && st->pts_buffer[i] == AV_NOPTS_VALUE; i++)

            st->pts_buffer[i]= pkt->pts + (i-delay-1) * pkt->duration;

        for(i=0; i<delay && st->pts_buffer[i] > st->pts_buffer[i+1]; i++)

            FFSWAP(int64_t, st->pts_buffer[i], st->pts_buffer[i+1]);



        pkt->dts= st->pts_buffer[0];

    }



    if(st->cur_dts && st->cur_dts != AV_NOPTS_VALUE && ((!(s->oformat->flags & AVFMT_TS_NONSTRICT) && st->cur_dts >= pkt->dts) || st->cur_dts > pkt->dts)){

        av_log(s, AV_LOG_ERROR,

               "Application provided invalid, non monotonically increasing dts to muxer in stream %d: %"PRId64" >= %"PRId64"\n",

               st->index, st->cur_dts, pkt->dts);

        return AVERROR(EINVAL);

    }

    if(pkt->dts != AV_NOPTS_VALUE && pkt->pts != AV_NOPTS_VALUE && pkt->pts < pkt->dts){

        av_log(s, AV_LOG_ERROR, "pts < dts in stream %d\n", st->index);

        return AVERROR(EINVAL);

    }



//    av_log(s, AV_LOG_DEBUG, "av_write_frame: pts2:%"PRId64" dts2:%"PRId64"\n", pkt->pts, pkt->dts);

    st->cur_dts= pkt->dts;

    st->pts.val= pkt->dts;



    /* update pts */

    switch (st->codec->codec_type) {

    case AVMEDIA_TYPE_AUDIO:

        frame_size = get_audio_frame_size(st->codec, pkt->size);



        /* HACK/FIXME, we skip the initial 0 size packets as they are most

           likely equal to the encoder delay, but it would be better if we

           had the real timestamps from the encoder */

        if (frame_size >= 0 && (pkt->size || st->pts.num!=st->pts.den>>1 || st->pts.val)) {

            frac_add(&st->pts, (int64_t)st->time_base.den * frame_size);

        }

        break;

    case AVMEDIA_TYPE_VIDEO:

        frac_add(&st->pts, (int64_t)st->time_base.den * st->codec->time_base.num);

        break;

    default:

        break;

    }

    return 0;

}
