rdt_parse_packet (AVFormatContext *ctx, PayloadContext *rdt, AVStream *st,

                  AVPacket *pkt, uint32_t *timestamp,

                  const uint8_t *buf, int len, uint16_t rtp_seq, int flags)

{

    int seq = 1, res;

    AVIOContext pb;



    if (!rdt->rmctx)

        return AVERROR(EINVAL);



    if (rdt->audio_pkt_cnt == 0) {

        int pos;



        ffio_init_context(&pb, buf, len, 0, NULL, NULL, NULL, NULL);

        flags = (flags & RTP_FLAG_KEY) ? 2 : 0;

        res = ff_rm_parse_packet (rdt->rmctx, &pb, st, rdt->rmst[st->index], len, pkt,

                                  &seq, flags, *timestamp);

        pos = avio_tell(&pb);

        if (res < 0)

            return res;

        if (res > 0) {

            if (st->codec->codec_id == AV_CODEC_ID_AAC) {

                memcpy (rdt->buffer, buf + pos, len - pos);

                rdt->rmctx->pb = avio_alloc_context (rdt->buffer, len - pos, 0,

                                                    NULL, NULL, NULL, NULL);

            }

            goto get_cache;

        }

    } else {

get_cache:

        rdt->audio_pkt_cnt =

            ff_rm_retrieve_cache (rdt->rmctx, rdt->rmctx->pb,

                                  st, rdt->rmst[st->index], pkt);

        if (rdt->audio_pkt_cnt == 0 &&

            st->codec->codec_id == AV_CODEC_ID_AAC)

            av_freep(&rdt->rmctx->pb);

    }

    pkt->stream_index = st->index;

    pkt->pts = *timestamp;



    return rdt->audio_pkt_cnt > 0;

}
