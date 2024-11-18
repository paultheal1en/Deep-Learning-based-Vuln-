static int vobsub_read_packet(AVFormatContext *s, AVPacket *pkt)

{

    MpegDemuxContext *vobsub = s->priv_data;

    FFDemuxSubtitlesQueue *q;

    AVIOContext *pb = vobsub->sub_ctx->pb;

    int ret, psize, total_read = 0, i;

    AVPacket idx_pkt;



    int64_t min_ts = INT64_MAX;

    int sid = 0;

    for (i = 0; i < s->nb_streams; i++) {

        FFDemuxSubtitlesQueue *tmpq = &vobsub->q[i];

        int64_t ts = tmpq->subs[tmpq->current_sub_idx].pts;

        if (ts < min_ts) {

            min_ts = ts;

            sid = i;

        }

    }

    q = &vobsub->q[sid];

    ret = ff_subtitles_queue_read_packet(q, &idx_pkt);

    if (ret < 0)

        return ret;



    /* compute maximum packet size using the next packet position. This is

     * useful when the len in the header is non-sense */

    if (q->current_sub_idx < q->nb_subs) {

        psize = q->subs[q->current_sub_idx].pos - idx_pkt.pos;

    } else {

        int64_t fsize = avio_size(pb);

        psize = fsize < 0 ? 0xffff : fsize - idx_pkt.pos;

    }



    avio_seek(pb, idx_pkt.pos, SEEK_SET);



    av_init_packet(pkt);

    pkt->size = 0;

    pkt->data = NULL;



    do {

        int n, to_read, startcode;

        int64_t pts, dts;

        int64_t old_pos = avio_tell(pb), new_pos;

        int pkt_size;



        ret = mpegps_read_pes_header(vobsub->sub_ctx, NULL, &startcode, &pts, &dts);

        if (ret < 0) {

            if (pkt->size) // raise packet even if incomplete

                break;

            goto fail;

        }

        to_read = ret & 0xffff;

        new_pos = avio_tell(pb);

        pkt_size = ret + (new_pos - old_pos);



        /* this prevents reads above the current packet */

        if (total_read + pkt_size > psize)

            break;

        total_read += pkt_size;



        /* the current chunk doesn't match the stream index (unlikely) */

        if ((startcode & 0x1f) != idx_pkt.stream_index)

            break;



        ret = av_grow_packet(pkt, to_read);

        if (ret < 0)

            goto fail;



        n = avio_read(pb, pkt->data + (pkt->size - to_read), to_read);

        if (n < to_read)

            pkt->size -= to_read - n;

    } while (total_read < psize);



    pkt->pts = pkt->dts = idx_pkt.pts;

    pkt->pos = idx_pkt.pos;

    pkt->stream_index = idx_pkt.stream_index;



    av_free_packet(&idx_pkt);

    return 0;



fail:

    av_free_packet(pkt);

    av_free_packet(&idx_pkt);

    return ret;

}
