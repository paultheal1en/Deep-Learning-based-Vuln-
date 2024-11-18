static int aea_read_header(AVFormatContext *s)

{

    AVStream *st = avformat_new_stream(s, NULL);

    if (!st)

        return AVERROR(ENOMEM);



    /* Parse the amount of channels and skip to pos 2048(0x800) */

    avio_skip(s->pb, 264);

    st->codec->channels = avio_r8(s->pb);

    avio_skip(s->pb, 1783);





    st->codec->codec_type     = AVMEDIA_TYPE_AUDIO;

    st->codec->codec_id       = AV_CODEC_ID_ATRAC1;

    st->codec->sample_rate    = 44100;

    st->codec->bit_rate       = 292000;



    if (st->codec->channels != 1 && st->codec->channels != 2) {

        av_log(s,AV_LOG_ERROR,"Channels %d not supported!\n",st->codec->channels);

        return -1;

    }



    st->codec->channel_layout = (st->codec->channels == 1) ? AV_CH_LAYOUT_MONO : AV_CH_LAYOUT_STEREO;



    st->codec->block_align = AT1_SU_SIZE * st->codec->channels;

    return 0;

}
