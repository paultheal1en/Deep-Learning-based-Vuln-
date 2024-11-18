static int vid_read_packet(AVFormatContext *s,

                           AVPacket *pkt)

{

    BVID_DemuxContext *vid = s->priv_data;

    AVIOContext *pb = s->pb;

    unsigned char block_type;

    int audio_length;

    int ret_value;



    if(vid->is_finished || pb->eof_reached)

        return AVERROR(EIO);



    block_type = avio_r8(pb);

    switch(block_type){

        case PALETTE_BLOCK:

            avio_seek(pb, -1, SEEK_CUR);     // include block type

            ret_value = av_get_packet(pb, pkt, 3 * 256 + 1);

            if(ret_value != 3 * 256 + 1){

                av_free_packet(pkt);

                return AVERROR(EIO);

            }

            pkt->stream_index = 0;

            return ret_value;



        case FIRST_AUDIO_BLOCK:

            avio_rl16(pb);

            // soundblaster DAC used for sample rate, as on specification page (link above)

            s->streams[1]->codec->sample_rate = 1000000 / (256 - avio_r8(pb));

            s->streams[1]->codec->bit_rate = s->streams[1]->codec->channels * s->streams[1]->codec->sample_rate * s->streams[1]->codec->bits_per_coded_sample;

        case AUDIO_BLOCK:

            audio_length = avio_rl16(pb);

            ret_value = av_get_packet(pb, pkt, audio_length);

            pkt->stream_index = 1;

            return ret_value != audio_length ? AVERROR(EIO) : ret_value;



        case VIDEO_P_FRAME:

        case VIDEO_YOFF_P_FRAME:

        case VIDEO_I_FRAME:

            return read_frame(vid, pb, pkt, block_type, s,

                              s->streams[0]->codec->width * s->streams[0]->codec->height);



        case EOF_BLOCK:

            if(vid->nframes != 0)

                av_log(s, AV_LOG_VERBOSE, "reached terminating character but not all frames read.\n");

            vid->is_finished = 1;

            return AVERROR(EIO);

        default:

            av_log(s, AV_LOG_ERROR, "unknown block (character = %c, decimal = %d, hex = %x)!!!\n",

                   block_type, block_type, block_type); return -1;

    }

}
