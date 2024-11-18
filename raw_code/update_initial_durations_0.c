static void update_initial_durations(AVFormatContext *s, AVStream *st,

                                     int stream_index, int duration)

{

    AVPacketList *pktl = s->internal->packet_buffer ? s->internal->packet_buffer : s->internal->parse_queue;

    int64_t cur_dts    = RELATIVE_TS_BASE;



    if (st->first_dts != AV_NOPTS_VALUE) {

        if (st->update_initial_durations_done)

            return;

        st->update_initial_durations_done = 1;

        cur_dts = st->first_dts;

        for (; pktl; pktl = get_next_pkt(s, st, pktl)) {

            if (pktl->pkt.stream_index == stream_index) {

                if (pktl->pkt.pts != pktl->pkt.dts  ||

                    pktl->pkt.dts != AV_NOPTS_VALUE ||

                    pktl->pkt.duration)

                    break;

                cur_dts -= duration;

            }

        }

        if (pktl && pktl->pkt.dts != st->first_dts) {

            av_log(s, AV_LOG_DEBUG, "first_dts %s not matching first dts %s (pts %s, duration %"PRId64") in the queue\n",

                   av_ts2str(st->first_dts), av_ts2str(pktl->pkt.dts), av_ts2str(pktl->pkt.pts), pktl->pkt.duration);

            return;

        }

        if (!pktl) {

            av_log(s, AV_LOG_DEBUG, "first_dts %s but no packet with dts in the queue\n", av_ts2str(st->first_dts));

            return;

        }

        pktl          = s->internal->packet_buffer ? s->internal->packet_buffer : s->internal->parse_queue;

        st->first_dts = cur_dts;

    } else if (st->cur_dts != RELATIVE_TS_BASE)

        return;



    for (; pktl; pktl = get_next_pkt(s, st, pktl)) {

        if (pktl->pkt.stream_index != stream_index)

            continue;

        if (pktl->pkt.pts == pktl->pkt.dts  &&

            (pktl->pkt.dts == AV_NOPTS_VALUE || pktl->pkt.dts == st->first_dts) &&

            !pktl->pkt.duration) {

            pktl->pkt.dts = cur_dts;

            if (!st->internal->avctx->has_b_frames)

                pktl->pkt.pts = cur_dts;

//            if (st->codecpar->codec_type != AVMEDIA_TYPE_AUDIO)

                pktl->pkt.duration = duration;

        } else

            break;

        cur_dts = pktl->pkt.dts + pktl->pkt.duration;

    }

    if (!pktl)

        st->cur_dts = cur_dts;

}
