static int wsvqa_read_packet(AVFormatContext *s,

                             AVPacket *pkt)

{

    WsVqaDemuxContext *wsvqa = s->priv_data;

    AVIOContext *pb = s->pb;

    int ret = -1;

    unsigned char preamble[VQA_PREAMBLE_SIZE];

    unsigned int chunk_type;

    unsigned int chunk_size;

    int skip_byte;



    while (avio_read(pb, preamble, VQA_PREAMBLE_SIZE) == VQA_PREAMBLE_SIZE) {

        chunk_type = AV_RB32(&preamble[0]);

        chunk_size = AV_RB32(&preamble[4]);



        skip_byte = chunk_size & 0x01;



        if ((chunk_type == SND0_TAG) || (chunk_type == SND1_TAG) ||

            (chunk_type == SND2_TAG) || (chunk_type == VQFR_TAG)) {



            ret= av_get_packet(pb, pkt, chunk_size);

            if (ret<0)

                return AVERROR(EIO);



            switch (chunk_type) {

            case SND0_TAG:

            case SND1_TAG:

            case SND2_TAG:

                if (wsvqa->audio_stream_index == -1) {

                    AVStream *st = avformat_new_stream(s, NULL);

                    if (!st)

                        return AVERROR(ENOMEM);



                    wsvqa->audio_stream_index = st->index;

                    if (!wsvqa->sample_rate)

                        wsvqa->sample_rate = 22050;

                    if (!wsvqa->channels)

                        wsvqa->channels = 1;

                    if (!wsvqa->bps)

                        wsvqa->bps = 8;

                    st->codec->sample_rate = wsvqa->sample_rate;

                    st->codec->bits_per_coded_sample = wsvqa->bps;

                    st->codec->channels = wsvqa->channels;

                    st->codec->codec_type = AVMEDIA_TYPE_AUDIO;



                    avpriv_set_pts_info(st, 64, 1, st->codec->sample_rate);



                    switch (chunk_type) {

                    case SND0_TAG:

                        if (wsvqa->bps == 16)

                            st->codec->codec_id = AV_CODEC_ID_PCM_S16LE;

                        else

                            st->codec->codec_id = AV_CODEC_ID_PCM_U8;

                        break;

                    case SND1_TAG:

                        st->codec->codec_id = AV_CODEC_ID_WESTWOOD_SND1;

                        break;

                    case SND2_TAG:

                        st->codec->codec_id = AV_CODEC_ID_ADPCM_IMA_WS;

                        st->codec->extradata_size = 2;

                        st->codec->extradata = av_mallocz(2 + FF_INPUT_BUFFER_PADDING_SIZE);

                        if (!st->codec->extradata)

                            return AVERROR(ENOMEM);

                        AV_WL16(st->codec->extradata, wsvqa->version);

                        break;

                    }

                }



                pkt->stream_index = wsvqa->audio_stream_index;

                switch (chunk_type) {

                case SND1_TAG:

                    /* unpacked size is stored in header */

                    pkt->duration = AV_RL16(pkt->data) / wsvqa->channels;

                    break;

                case SND2_TAG:

                    /* 2 samples/byte, 1 or 2 samples per frame depending on stereo */

                    pkt->duration = (chunk_size * 2) / wsvqa->channels;

                    break;

                }

                break;

            case VQFR_TAG:

                pkt->stream_index = wsvqa->video_stream_index;

                pkt->duration = 1;

                break;

            }



            /* stay on 16-bit alignment */

            if (skip_byte)

                avio_skip(pb, 1);



            return ret;

        } else {

            switch(chunk_type){

            case CMDS_TAG:

                break;

            default:

                av_log(s, AV_LOG_INFO, "Skipping unknown chunk 0x%08X\n", chunk_type);

            }

            avio_skip(pb, chunk_size + skip_byte);

        }

    }



    return ret;

}
