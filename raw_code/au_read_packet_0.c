static int au_read_packet(AVFormatContext *s,

                          AVPacket *pkt)

{

    int ret;



    ret= av_get_packet(s->pb, pkt, BLOCK_SIZE *

                       s->streams[0]->codec->channels *

                       av_get_bits_per_sample(s->streams[0]->codec->codec_id) >> 3);

    if (ret < 0)

        return ret;

    pkt->flags &= ~AV_PKT_FLAG_CORRUPT;

    pkt->stream_index = 0;

    return 0;

}
