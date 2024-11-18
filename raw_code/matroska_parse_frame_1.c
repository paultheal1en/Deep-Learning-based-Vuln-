static int matroska_parse_frame(MatroskaDemuxContext *matroska,

                                MatroskaTrack *track, AVStream *st,

                                uint8_t *data, int pkt_size,

                                uint64_t timecode, uint64_t duration,

                                int64_t pos, int is_keyframe)

{

    MatroskaTrackEncoding *encodings = track->encodings.elem;

    uint8_t *pkt_data = data;

    int offset = 0, res;

    AVPacket *pkt;



    if (encodings && encodings->scope & 1) {

        res = matroska_decode_buffer(&pkt_data, &pkt_size, track);

        if (res < 0)

            return res;

    }



    if (st->codec->codec_id == AV_CODEC_ID_WAVPACK) {

        uint8_t *wv_data;

        res = matroska_parse_wavpack(track, pkt_data, &wv_data, &pkt_size);

        if (res < 0) {

            av_log(matroska->ctx, AV_LOG_ERROR,

                   "Error parsing a wavpack block.\n");

            goto fail;

        }

        if (pkt_data != data)


        pkt_data = wv_data;

    }



    if (st->codec->codec_id == AV_CODEC_ID_PRORES)

        offset = 8;



    pkt = av_mallocz(sizeof(AVPacket));

    /* XXX: prevent data copy... */

    if (av_new_packet(pkt, pkt_size + offset) < 0) {

        av_free(pkt);


        return AVERROR(ENOMEM);

    }



    if (st->codec->codec_id == AV_CODEC_ID_PRORES) {

        uint8_t *buf = pkt->data;

        bytestream_put_be32(&buf, pkt_size);

        bytestream_put_be32(&buf, MKBETAG('i', 'c', 'p', 'f'));

    }



    memcpy(pkt->data + offset, pkt_data, pkt_size);



    if (pkt_data != data)

        av_free(pkt_data);



    pkt->flags        = is_keyframe;

    pkt->stream_index = st->index;



    if (track->ms_compat)

        pkt->dts = timecode;

    else

        pkt->pts = timecode;

    pkt->pos = pos;

    if (st->codec->codec_id == AV_CODEC_ID_TEXT)

        pkt->convergence_duration = duration;

    else if (track->type != MATROSKA_TRACK_TYPE_SUBTITLE)

        pkt->duration = duration;



    if (st->codec->codec_id == AV_CODEC_ID_SSA)

        matroska_fix_ass_packet(matroska, pkt, duration);



    if (matroska->prev_pkt                                 &&

        timecode                         != AV_NOPTS_VALUE &&

        matroska->prev_pkt->pts          == timecode       &&

        matroska->prev_pkt->stream_index == st->index      &&

        st->codec->codec_id == AV_CODEC_ID_SSA)

        matroska_merge_packets(matroska->prev_pkt, pkt);

    else {

        dynarray_add(&matroska->packets, &matroska->num_packets, pkt);

        matroska->prev_pkt = pkt;

    }



    return 0;



fail:

    if (pkt_data != data)


    return res;

}