static int mpegts_write_packet(AVFormatContext *s, AVPacket *pkt)

{

    AVStream *st = s->streams[pkt->stream_index];

    int size= pkt->size;

    uint8_t *buf= pkt->data;

    MpegTSWriteStream *ts_st = st->priv_data;

    int len, max_payload_size;

    const uint8_t *access_unit_index = NULL;



    if (st->codec->codec_type == CODEC_TYPE_SUBTITLE) {

        /* for subtitle, a single PES packet must be generated */

        mpegts_write_pes(s, st, buf, size, pkt->pts, AV_NOPTS_VALUE);

        return 0;

    }



    if (st->codec->codec_id == CODEC_ID_DIRAC) {

        /* for Dirac, a single PES packet must be generated */

        mpegts_write_pes(s, st, buf, size, pkt->pts, pkt->dts);

        return 0;

    }

    max_payload_size = DEFAULT_PES_PAYLOAD_SIZE;

    if (st->codec->codec_id == CODEC_ID_MPEG2VIDEO ||

        st->codec->codec_id == CODEC_ID_MPEG1VIDEO) {

        const uint8_t *p = pkt->data;

        const uint8_t *end = pkt->data+pkt->size;

        uint32_t state = -1;

        while (p < end) {

            p = ff_find_start_code(p, end, &state);

            if (state == PICTURE_START_CODE) {

                access_unit_index = p - 4;

                break;

            }

        }

    } else if (st->codec->codec_type == CODEC_TYPE_AUDIO) {

        access_unit_index = pkt->data;

    }



    if (!access_unit_index) {

        av_log(s, AV_LOG_ERROR, "error, could not find access unit start\n");

        return -1;

    }



    while (size > 0) {

        len = max_payload_size - ts_st->payload_index;

        if (len > size)

            len = size;

        memcpy(ts_st->payload + ts_st->payload_index, buf, len);

        buf += len;

        size -= len;

        ts_st->payload_index += len;

        if (access_unit_index && access_unit_index < buf &&

            ts_st->payload_pts == AV_NOPTS_VALUE &&

            ts_st->payload_dts == AV_NOPTS_VALUE) {

            ts_st->payload_dts = pkt->dts;

            ts_st->payload_pts = pkt->pts;

        }

        if (ts_st->payload_index >= max_payload_size) {

            mpegts_write_pes(s, st, ts_st->payload, ts_st->payload_index,

                             ts_st->payload_pts, ts_st->payload_dts);

            ts_st->payload_pts = AV_NOPTS_VALUE;

            ts_st->payload_dts = AV_NOPTS_VALUE;

            ts_st->payload_index = 0;

            access_unit_index = NULL; // unset access unit to avoid setting pts/dts again

        }

    }

    return 0;

}
