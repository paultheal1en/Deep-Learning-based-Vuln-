static int parse_vtrk(AVFormatContext *s,

                      FourxmDemuxContext *fourxm, uint8_t *buf, int size)

{

    AVStream *st;

    /* check that there is enough data */

    if (size != vtrk_SIZE) {

        return AVERROR_INVALIDDATA;

    }



    /* allocate a new AVStream */

    st = avformat_new_stream(s, NULL);

    if (!st)

        return AVERROR(ENOMEM);



    avpriv_set_pts_info(st, 60, 1, fourxm->fps);



    fourxm->video_stream_index = st->index;



    st->codec->codec_type     = AVMEDIA_TYPE_VIDEO;

    st->codec->codec_id       = AV_CODEC_ID_4XM;

    st->codec->extradata_size = 4;

    st->codec->extradata      = av_malloc(4);

    AV_WL32(st->codec->extradata, AV_RL32(buf + 16));

    st->codec->width  = AV_RL32(buf + 36);

    st->codec->height = AV_RL32(buf + 40);



    return 0;

}
