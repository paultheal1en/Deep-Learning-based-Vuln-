static void compute_pkt_fields(AVFormatContext *s, AVStream *st, 

                               AVCodecParserContext *pc, AVPacket *pkt)

{

    int num, den, presentation_delayed;



    /* handle wrapping */

    if(st->cur_dts != AV_NOPTS_VALUE){

        if(pkt->pts != AV_NOPTS_VALUE)

            pkt->pts= lsb2full(pkt->pts, st->cur_dts, st->pts_wrap_bits);

        if(pkt->dts != AV_NOPTS_VALUE)

            pkt->dts= lsb2full(pkt->dts, st->cur_dts, st->pts_wrap_bits);

    }

    

    if (pkt->duration == 0) {

        compute_frame_duration(&num, &den, s, st, pc, pkt);

        if (den && num) {

            pkt->duration = av_rescale(1, num * (int64_t)st->time_base.den, den * (int64_t)st->time_base.num);

        }

    }



    /* do we have a video B frame ? */

    presentation_delayed = 0;

    if (st->codec.codec_type == CODEC_TYPE_VIDEO) {

        /* XXX: need has_b_frame, but cannot get it if the codec is

           not initialized */

        if ((st->codec.codec_id == CODEC_ID_MPEG1VIDEO ||

             st->codec.codec_id == CODEC_ID_MPEG2VIDEO ||

             st->codec.codec_id == CODEC_ID_MPEG4 ||

             st->codec.codec_id == CODEC_ID_H264) && 

            pc && pc->pict_type != FF_B_TYPE)

            presentation_delayed = 1;

        /* this may be redundant, but it shouldnt hurt */

        if(pkt->dts != AV_NOPTS_VALUE && pkt->pts != AV_NOPTS_VALUE && pkt->pts > pkt->dts)

            presentation_delayed = 1;

    }

    

    if(st->cur_dts == AV_NOPTS_VALUE){

        if(presentation_delayed) st->cur_dts = -pkt->duration;

        else                     st->cur_dts = 0;

    }



//    av_log(NULL, AV_LOG_DEBUG, "IN delayed:%d pts:%lld, dts:%lld cur_dts:%lld\n", presentation_delayed, pkt->pts, pkt->dts, st->cur_dts);

    /* interpolate PTS and DTS if they are not present */

    if (presentation_delayed) {

        /* DTS = decompression time stamp */

        /* PTS = presentation time stamp */

        if (pkt->dts == AV_NOPTS_VALUE) {

            /* if we know the last pts, use it */

            if(st->last_IP_pts != AV_NOPTS_VALUE)

                st->cur_dts = pkt->dts = st->last_IP_pts;

            else

                pkt->dts = st->cur_dts;

        } else {

            st->cur_dts = pkt->dts;

        }

        /* this is tricky: the dts must be incremented by the duration

           of the frame we are displaying, i.e. the last I or P frame */

        if (st->last_IP_duration == 0)

            st->cur_dts += pkt->duration;

        else

            st->cur_dts += st->last_IP_duration;

        st->last_IP_duration  = pkt->duration;

        st->last_IP_pts= pkt->pts;

        /* cannot compute PTS if not present (we can compute it only

           by knowing the futur */

    } else {

        /* presentation is not delayed : PTS and DTS are the same */

        if (pkt->pts == AV_NOPTS_VALUE) {

            if (pkt->dts == AV_NOPTS_VALUE) {

                pkt->pts = st->cur_dts;

                pkt->dts = st->cur_dts;

            }

            else {

                st->cur_dts = pkt->dts;

                pkt->pts = pkt->dts;

            }

        } else {

            st->cur_dts = pkt->pts;

            pkt->dts = pkt->pts;

        }

        st->cur_dts += pkt->duration;

    }

//    av_log(NULL, AV_LOG_DEBUG, "OUTdelayed:%d pts:%lld, dts:%lld cur_dts:%lld\n", presentation_delayed, pkt->pts, pkt->dts, st->cur_dts);

    

    /* update flags */

    if (pc) {

        pkt->flags = 0;

        /* key frame computation */

        switch(st->codec.codec_type) {

        case CODEC_TYPE_VIDEO:

            if (pc->pict_type == FF_I_TYPE)

                pkt->flags |= PKT_FLAG_KEY;

            break;

        case CODEC_TYPE_AUDIO:

            pkt->flags |= PKT_FLAG_KEY;

            break;

        default:

            break;

        }

    }



    /* convert the packet time stamp units */

    if(pkt->pts != AV_NOPTS_VALUE)

        pkt->pts = av_rescale(pkt->pts, AV_TIME_BASE * (int64_t)st->time_base.num, st->time_base.den);

    if(pkt->dts != AV_NOPTS_VALUE)

        pkt->dts = av_rescale(pkt->dts, AV_TIME_BASE * (int64_t)st->time_base.num, st->time_base.den);



    /* duration field */

    pkt->duration = av_rescale(pkt->duration, AV_TIME_BASE * (int64_t)st->time_base.num, st->time_base.den);

}
