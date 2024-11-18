ff_rm_retrieve_cache (AVFormatContext *s, AVIOContext *pb,

                      AVStream *st, RMStream *ast, AVPacket *pkt)

{

    RMDemuxContext *rm = s->priv_data;



    assert (rm->audio_pkt_cnt > 0);



    if (st->codec->codec_id == CODEC_ID_AAC)

        av_get_packet(pb, pkt, ast->sub_packet_lengths[ast->sub_packet_cnt - rm->audio_pkt_cnt]);

    else {

        av_new_packet(pkt, st->codec->block_align);

        memcpy(pkt->data, ast->pkt.data + st->codec->block_align * //FIXME avoid this

               (ast->sub_packet_h * ast->audio_framesize / st->codec->block_align - rm->audio_pkt_cnt),

               st->codec->block_align);

    }

    rm->audio_pkt_cnt--;

    if ((pkt->pts = ast->audiotimestamp) != AV_NOPTS_VALUE) {

        ast->audiotimestamp = AV_NOPTS_VALUE;

        pkt->flags = AV_PKT_FLAG_KEY;

    } else

        pkt->flags = 0;

    pkt->stream_index = st->index;



    return rm->audio_pkt_cnt;

}
