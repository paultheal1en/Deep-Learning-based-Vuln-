static int wsvqa_read_header(AVFormatContext *s,

                             AVFormatParameters *ap)

{

    WsVqaDemuxContext *wsvqa = s->priv_data;

    AVIOContext *pb = s->pb;

    AVStream *st;

    unsigned char *header;

    unsigned char scratch[VQA_PREAMBLE_SIZE];

    unsigned int chunk_tag;

    unsigned int chunk_size;



    /* initialize the video decoder stream */

    st = av_new_stream(s, 0);

    if (!st)

        return AVERROR(ENOMEM);

    av_set_pts_info(st, 33, 1, VQA_FRAMERATE);

    wsvqa->video_stream_index = st->index;

    st->codec->codec_type = AVMEDIA_TYPE_VIDEO;

    st->codec->codec_id = CODEC_ID_WS_VQA;

    st->codec->codec_tag = 0;  /* no fourcc */



    /* skip to the start of the VQA header */

    avio_seek(pb, 20, SEEK_SET);



    /* the VQA header needs to go to the decoder */

    st->codec->extradata_size = VQA_HEADER_SIZE;

    st->codec->extradata = av_mallocz(VQA_HEADER_SIZE + FF_INPUT_BUFFER_PADDING_SIZE);

    header = (unsigned char *)st->codec->extradata;

    if (avio_read(pb, st->codec->extradata, VQA_HEADER_SIZE) !=

        VQA_HEADER_SIZE) {

        av_free(st->codec->extradata);

        return AVERROR(EIO);

    }

    st->codec->width = AV_RL16(&header[6]);

    st->codec->height = AV_RL16(&header[8]);



    /* initialize the audio decoder stream for VQA v1 or nonzero samplerate */

    if (AV_RL16(&header[24]) || (AV_RL16(&header[0]) == 1 && AV_RL16(&header[2]) == 1)) {

        st = av_new_stream(s, 0);

        if (!st)

            return AVERROR(ENOMEM);

        av_set_pts_info(st, 33, 1, VQA_FRAMERATE);

        st->codec->codec_type = AVMEDIA_TYPE_AUDIO;

        if (AV_RL16(&header[0]) == 1)

            st->codec->codec_id = CODEC_ID_WESTWOOD_SND1;

        else

            st->codec->codec_id = CODEC_ID_ADPCM_IMA_WS;

        st->codec->codec_tag = 0;  /* no tag */

        st->codec->sample_rate = AV_RL16(&header[24]);

        if (!st->codec->sample_rate)

            st->codec->sample_rate = 22050;

        st->codec->channels = header[26];

        if (!st->codec->channels)

            st->codec->channels = 1;

        st->codec->bits_per_coded_sample = 16;

        st->codec->bit_rate = st->codec->channels * st->codec->sample_rate *

            st->codec->bits_per_coded_sample / 4;

        st->codec->block_align = st->codec->channels * st->codec->bits_per_coded_sample;



        wsvqa->audio_stream_index = st->index;

        wsvqa->audio_samplerate = st->codec->sample_rate;

        wsvqa->audio_channels = st->codec->channels;

        wsvqa->audio_frame_counter = 0;

    }



    /* there are 0 or more chunks before the FINF chunk; iterate until

     * FINF has been skipped and the file will be ready to be demuxed */

    do {

        if (avio_read(pb, scratch, VQA_PREAMBLE_SIZE) != VQA_PREAMBLE_SIZE) {

            av_free(st->codec->extradata);

            return AVERROR(EIO);

        }

        chunk_tag = AV_RB32(&scratch[0]);

        chunk_size = AV_RB32(&scratch[4]);



        /* catch any unknown header tags, for curiousity */

        switch (chunk_tag) {

        case CINF_TAG:

        case CINH_TAG:

        case CIND_TAG:

        case PINF_TAG:

        case PINH_TAG:

        case PIND_TAG:

        case FINF_TAG:

        case CMDS_TAG:

            break;



        default:

            av_log (s, AV_LOG_ERROR, " note: unknown chunk seen (%c%c%c%c)\n",

                scratch[0], scratch[1],

                scratch[2], scratch[3]);

            break;

        }



        avio_skip(pb, chunk_size);

    } while (chunk_tag != FINF_TAG);



    return 0;

}
