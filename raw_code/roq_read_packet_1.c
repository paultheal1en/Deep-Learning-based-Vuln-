static int roq_read_packet(AVFormatContext *s,

                           AVPacket *pkt)

{

    RoqDemuxContext *roq = s->priv_data;

    AVIOContext *pb = s->pb;

    int ret = 0;

    unsigned int chunk_size;

    unsigned int chunk_type;

    unsigned int codebook_size;

    unsigned char preamble[RoQ_CHUNK_PREAMBLE_SIZE];

    int packet_read = 0;

    int64_t codebook_offset;



    while (!packet_read) {



        if (avio_feof(s->pb))

            return AVERROR(EIO);



        /* get the next chunk preamble */

        if ((ret = avio_read(pb, preamble, RoQ_CHUNK_PREAMBLE_SIZE)) !=

            RoQ_CHUNK_PREAMBLE_SIZE)

            return AVERROR(EIO);



        chunk_type = AV_RL16(&preamble[0]);

        chunk_size = AV_RL32(&preamble[2]);

        if(chunk_size > INT_MAX)

            return AVERROR_INVALIDDATA;



        chunk_size = ffio_limit(pb, chunk_size);



        switch (chunk_type) {



        case RoQ_INFO:

            if (roq->video_stream_index == -1) {

                AVStream *st = avformat_new_stream(s, NULL);

                if (!st)

                    return AVERROR(ENOMEM);

                avpriv_set_pts_info(st, 63, 1, roq->frame_rate);

                roq->video_stream_index = st->index;

                st->codecpar->codec_type   = AVMEDIA_TYPE_VIDEO;

                st->codecpar->codec_id     = AV_CODEC_ID_ROQ;

                st->codecpar->codec_tag    = 0;  /* no fourcc */



                if (avio_read(pb, preamble, RoQ_CHUNK_PREAMBLE_SIZE) != RoQ_CHUNK_PREAMBLE_SIZE)

                    return AVERROR(EIO);

                st->codecpar->width  = roq->width  = AV_RL16(preamble);

                st->codecpar->height = roq->height = AV_RL16(preamble + 2);

                break;

            }

            /* don't care about this chunk anymore */

            avio_skip(pb, RoQ_CHUNK_PREAMBLE_SIZE);

            break;



        case RoQ_QUAD_CODEBOOK:

            if (roq->video_stream_index < 0)

                return AVERROR_INVALIDDATA;

            /* packet needs to contain both this codebook and next VQ chunk */

            codebook_offset = avio_tell(pb) - RoQ_CHUNK_PREAMBLE_SIZE;

            codebook_size = chunk_size;

            avio_skip(pb, codebook_size);

            if (avio_read(pb, preamble, RoQ_CHUNK_PREAMBLE_SIZE) !=

                RoQ_CHUNK_PREAMBLE_SIZE)

                return AVERROR(EIO);

            chunk_size = AV_RL32(&preamble[2]) + RoQ_CHUNK_PREAMBLE_SIZE * 2 +

                codebook_size;



            if (chunk_size > INT_MAX)

                return AVERROR_INVALIDDATA;



            /* rewind */

            avio_seek(pb, codebook_offset, SEEK_SET);



            /* load up the packet */

            ret= av_get_packet(pb, pkt, chunk_size);

            if (ret != chunk_size)

                return AVERROR(EIO);

            pkt->stream_index = roq->video_stream_index;

            pkt->pts = roq->video_pts++;



            packet_read = 1;

            break;



        case RoQ_SOUND_MONO:

        case RoQ_SOUND_STEREO:

            if (roq->audio_stream_index == -1) {

                AVStream *st = avformat_new_stream(s, NULL);

                if (!st)

                    return AVERROR(ENOMEM);

                avpriv_set_pts_info(st, 32, 1, RoQ_AUDIO_SAMPLE_RATE);

                roq->audio_stream_index = st->index;

                st->codecpar->codec_type = AVMEDIA_TYPE_AUDIO;

                st->codecpar->codec_id = AV_CODEC_ID_ROQ_DPCM;

                st->codecpar->codec_tag = 0;  /* no tag */

                if (chunk_type == RoQ_SOUND_STEREO) {

                    st->codecpar->channels       = 2;

                    st->codecpar->channel_layout = AV_CH_LAYOUT_STEREO;

                } else {

                    st->codecpar->channels       = 1;

                    st->codecpar->channel_layout = AV_CH_LAYOUT_MONO;

                }

                roq->audio_channels    = st->codecpar->channels;

                st->codecpar->sample_rate = RoQ_AUDIO_SAMPLE_RATE;

                st->codecpar->bits_per_coded_sample = 16;

                st->codecpar->bit_rate = st->codecpar->channels * st->codecpar->sample_rate *

                    st->codecpar->bits_per_coded_sample;

                st->codecpar->block_align = st->codecpar->channels * st->codecpar->bits_per_coded_sample;

            }

        case RoQ_QUAD_VQ:

            if (chunk_type == RoQ_QUAD_VQ) {

                if (roq->video_stream_index < 0)

                    return AVERROR_INVALIDDATA;

            }



            /* load up the packet */

            if (av_new_packet(pkt, chunk_size + RoQ_CHUNK_PREAMBLE_SIZE))

                return AVERROR(EIO);

            /* copy over preamble */

            memcpy(pkt->data, preamble, RoQ_CHUNK_PREAMBLE_SIZE);



            if (chunk_type == RoQ_QUAD_VQ) {

                pkt->stream_index = roq->video_stream_index;

                pkt->pts = roq->video_pts++;

            } else {

                pkt->stream_index = roq->audio_stream_index;

                pkt->pts = roq->audio_frame_count;

                roq->audio_frame_count += (chunk_size / roq->audio_channels);

            }



            pkt->pos= avio_tell(pb);

            ret = avio_read(pb, pkt->data + RoQ_CHUNK_PREAMBLE_SIZE,

                chunk_size);

            if (ret != chunk_size)

                ret = AVERROR(EIO);



            packet_read = 1;

            break;



        default:

            av_log(s, AV_LOG_ERROR, "  unknown RoQ chunk (%04X)\n", chunk_type);

            return AVERROR_INVALIDDATA;

        }

    }



    return ret;

}
