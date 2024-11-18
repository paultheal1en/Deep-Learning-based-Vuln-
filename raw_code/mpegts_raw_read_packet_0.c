static int mpegts_raw_read_packet(AVFormatContext *s,

                                  AVPacket *pkt)

{

    MpegTSContext *ts = s->priv_data;

    int ret, i;

    int64_t pcr_h, next_pcr_h, pos;

    int pcr_l, next_pcr_l;

    uint8_t pcr_buf[12];



    if (av_new_packet(pkt, TS_PACKET_SIZE) < 0)

        return AVERROR(ENOMEM);

    pkt->pos= url_ftell(s->pb);

    ret = read_packet(s->pb, pkt->data, ts->raw_packet_size);

    if (ret < 0) {

        av_free_packet(pkt);

        return ret;

    }

    if (ts->mpeg2ts_compute_pcr) {

        /* compute exact PCR for each packet */

        if (parse_pcr(&pcr_h, &pcr_l, pkt->data) == 0) {

            /* we read the next PCR (XXX: optimize it by using a bigger buffer */

            pos = url_ftell(s->pb);

            for(i = 0; i < MAX_PACKET_READAHEAD; i++) {

                url_fseek(s->pb, pos + i * ts->raw_packet_size, SEEK_SET);

                get_buffer(s->pb, pcr_buf, 12);

                if (parse_pcr(&next_pcr_h, &next_pcr_l, pcr_buf) == 0) {

                    /* XXX: not precise enough */

                    ts->pcr_incr = ((next_pcr_h - pcr_h) * 300 + (next_pcr_l - pcr_l)) /

                        (i + 1);

                    break;

                }

            }

            url_fseek(s->pb, pos, SEEK_SET);

            /* no next PCR found: we use previous increment */

            ts->cur_pcr = pcr_h * 300 + pcr_l;

        }

        pkt->pts = ts->cur_pcr;

        pkt->duration = ts->pcr_incr;

        ts->cur_pcr += ts->pcr_incr;

    }

    pkt->stream_index = 0;

    return 0;

}
