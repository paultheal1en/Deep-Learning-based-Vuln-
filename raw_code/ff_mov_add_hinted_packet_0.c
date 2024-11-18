int ff_mov_add_hinted_packet(AVFormatContext *s, AVPacket *pkt,

                             int track_index, int sample)

{

    MOVMuxContext *mov = s->priv_data;

    MOVTrack *trk = &mov->tracks[track_index];

    AVFormatContext *rtp_ctx = trk->rtp_ctx;

    uint8_t *buf = NULL;

    int size;

    AVIOContext *hintbuf = NULL;

    AVPacket hint_pkt;

    int ret = 0, count;



    if (!rtp_ctx)

        return AVERROR(ENOENT);

    if (!rtp_ctx->pb)

        return AVERROR(ENOMEM);



    sample_queue_push(&trk->sample_queue, pkt, sample);



    /* Feed the packet to the RTP muxer */

    ff_write_chained(rtp_ctx, 0, pkt, s);



    /* Fetch the output from the RTP muxer, open a new output buffer

     * for next time. */

    size = avio_close_dyn_buf(rtp_ctx->pb, &buf);

    if ((ret = url_open_dyn_packet_buf(&rtp_ctx->pb,

                                       RTP_MAX_PACKET_SIZE)) < 0)

        goto done;



    if (size <= 0)

        goto done;



    /* Open a buffer for writing the hint */

    if ((ret = avio_open_dyn_buf(&hintbuf)) < 0)

        goto done;

    av_init_packet(&hint_pkt);

    count = write_hint_packets(hintbuf, buf, size, trk, &hint_pkt.dts);

    av_freep(&buf);



    /* Write the hint data into the hint track */

    hint_pkt.size = size = avio_close_dyn_buf(hintbuf, &buf);

    hint_pkt.data = buf;

    hint_pkt.pts  = hint_pkt.dts;

    hint_pkt.stream_index = track_index;

    if (pkt->flags & AV_PKT_FLAG_KEY)

        hint_pkt.flags |= AV_PKT_FLAG_KEY;

    if (count > 0)

        ff_mov_write_packet(s, &hint_pkt);

done:

    av_free(buf);

    sample_queue_retain(&trk->sample_queue);

    return ret;

}
