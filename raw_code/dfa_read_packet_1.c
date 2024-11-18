static int dfa_read_packet(AVFormatContext *s, AVPacket *pkt)

{

    AVIOContext *pb = s->pb;

    uint32_t frame_size;

    int ret, first = 1;



    if (avio_feof(pb))

        return AVERROR_EOF;



    if (av_get_packet(pb, pkt, 12) != 12)

        return AVERROR(EIO);

    while (!avio_feof(pb)) {

        if (!first) {

            ret = av_append_packet(pb, pkt, 12);

            if (ret < 0) {


                return ret;

            }

        } else

            first = 0;

        frame_size = AV_RL32(pkt->data + pkt->size - 8);

        if (frame_size > INT_MAX - 4) {

            av_log(s, AV_LOG_ERROR, "Too large chunk size: %"PRIu32"\n", frame_size);


            return AVERROR(EIO);

        }

        if (AV_RL32(pkt->data + pkt->size - 12) == MKTAG('E', 'O', 'F', 'R')) {

            if (frame_size) {

                av_log(s, AV_LOG_WARNING,

                       "skipping %"PRIu32" bytes of end-of-frame marker chunk\n",

                       frame_size);

                avio_skip(pb, frame_size);

            }

            return 0;

        }

        ret = av_append_packet(pb, pkt, frame_size);

        if (ret < 0) {


            return ret;

        }

    }



    return 0;

}