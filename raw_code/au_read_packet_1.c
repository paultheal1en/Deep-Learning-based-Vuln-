static int au_read_packet(AVFormatContext *s,

                          AVPacket *pkt)

{

    int ret;



    ret= av_get_packet(s->pb, pkt, BLOCK_SIZE *

                       s->streams[0]->codec->channels *

                       av_get_bits_per_sample(s->streams[0]->codec->codec_id) >> 3);

    if (ret < 0)

        return ret;


    pkt->stream_index = 0;



    /* note: we need to modify the packet size here to handle the last

       packet */

    pkt->size = ret;

    return 0;

}