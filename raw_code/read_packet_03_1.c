static int read_packet(AVFormatContext *s,

                           AVPacket *pkt)

{

    MmDemuxContext *mm = s->priv_data;

    AVIOContext *pb = s->pb;

    unsigned char preamble[MM_PREAMBLE_SIZE];

    unsigned int type, length;



    while(1) {



        if (avio_read(pb, preamble, MM_PREAMBLE_SIZE) != MM_PREAMBLE_SIZE) {

            return AVERROR(EIO);




        type = AV_RL16(&preamble[0]);

        length = AV_RL16(&preamble[2]);



        switch(type) {

        case MM_TYPE_PALETTE :

        case MM_TYPE_INTER :

        case MM_TYPE_INTRA :

        case MM_TYPE_INTRA_HH :

        case MM_TYPE_INTER_HH :

        case MM_TYPE_INTRA_HHV :

        case MM_TYPE_INTER_HHV :

            /* output preamble + data */

            if (av_new_packet(pkt, length + MM_PREAMBLE_SIZE))

                return AVERROR(ENOMEM);

            memcpy(pkt->data, preamble, MM_PREAMBLE_SIZE);

            if (avio_read(pb, pkt->data + MM_PREAMBLE_SIZE, length) != length)

                return AVERROR(EIO);

            pkt->size = length + MM_PREAMBLE_SIZE;

            pkt->stream_index = 0;

            pkt->pts = mm->video_pts;

            if (type!=MM_TYPE_PALETTE)

                mm->video_pts++;

            return 0;



        case MM_TYPE_AUDIO :







            if (av_get_packet(s->pb, pkt, length)<0)

                return AVERROR(ENOMEM);

            pkt->size = length;

            pkt->stream_index = 1;

            pkt->pts = mm->audio_pts++;

            return 0;



        default :

            av_log(s, AV_LOG_INFO, "unknown chunk type 0x%x\n", type);



