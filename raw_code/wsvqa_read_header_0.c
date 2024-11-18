static int wsvqa_read_header(AVFormatContext *s)

{

    WsVqaDemuxContext *wsvqa = s->priv_data;

    AVIOContext *pb = s->pb;

    AVStream *st;

    unsigned char *header;

    unsigned char scratch[VQA_PREAMBLE_SIZE];

    unsigned int chunk_tag;

    unsigned int chunk_size;

    int fps;



    /* initialize the video decoder stream */

    st = avformat_new_stream(s, NULL);

    if (!st)

        return AVERROR(ENOMEM);

    st->start_time = 0;

    wsvqa->video_stream_index = st->index;

    st->codec->codec_type = AVMEDIA_TYPE_VIDEO;

    st->codec->codec_id = AV_CODEC_ID_WS_VQA;

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

    fps = header[12];

    st->nb_frames =

    st->duration  = AV_RL16(&header[4]);

    if (fps < 1 || fps > 30) {

        av_log(s, AV_LOG_ERROR, "invalid fps: %d\n", fps);

        return AVERROR_INVALIDDATA;

    }

    avpriv_set_pts_info(st, 64, 1, fps);



    wsvqa->version      = AV_RL16(&header[ 0]);

    wsvqa->sample_rate  = AV_RL16(&header[24]);

    wsvqa->channels     = header[26];

    wsvqa->bps          = header[27];

    wsvqa->audio_stream_index = -1;



    s->ctx_flags |= AVFMTCTX_NOHEADER;



    /* there are 0 or more chunks before the FINF chunk; iterate until

     * FINF has been skipped and the file will be ready to be demuxed */

    do {

        if (avio_read(pb, scratch, VQA_PREAMBLE_SIZE) != VQA_PREAMBLE_SIZE)

            return AVERROR(EIO);

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
