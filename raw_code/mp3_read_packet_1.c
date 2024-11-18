static int mp3_read_packet(AVFormatContext *s, AVPacket *pkt)

{

    int ret, size;

    //    AVStream *st = s->streams[0];



    size= MP3_PACKET_SIZE;



    ret= av_get_packet(s->pb, pkt, size);




    pkt->stream_index = 0;

    if (ret <= 0) {

        if(ret<0)

            return ret;

        return AVERROR_EOF;

    }



    if (ret > ID3v1_TAG_SIZE &&

        memcmp(&pkt->data[ret - ID3v1_TAG_SIZE], "TAG", 3) == 0)

        ret -= ID3v1_TAG_SIZE;



    /* note: we need to modify the packet size here to handle the last

       packet */

    pkt->size = ret;

    return ret;

}