static int parse_audio_var(AVFormatContext *avctx, AVStream *st, const char *name, int size)

{

    AVIOContext *pb = avctx->pb;

    if (!strcmp(name, "__DIR_COUNT")) {

        st->nb_frames = var_read_int(pb, size);

    } else if (!strcmp(name, "AUDIO_FORMAT")) {

        st->codec->codec_id = var_read_int(pb, size);

    } else if (!strcmp(name, "COMPRESSION")) {

        st->codec->codec_tag = var_read_int(pb, size);

    } else if (!strcmp(name, "DEFAULT_VOL")) {

        var_read_metadata(avctx, name, size);

    } else if (!strcmp(name, "NUM_CHANNELS")) {

        st->codec->channels = var_read_int(pb, size);

        st->codec->channel_layout = (st->codec->channels == 1) ? AV_CH_LAYOUT_MONO : AV_CH_LAYOUT_STEREO;

    } else if (!strcmp(name, "SAMPLE_RATE")) {

        st->codec->sample_rate = var_read_int(pb, size);

        avpriv_set_pts_info(st, 33, 1, st->codec->sample_rate);

    } else if (!strcmp(name, "SAMPLE_WIDTH")) {

        st->codec->bits_per_coded_sample = var_read_int(pb, size) * 8;

    } else

        return -1;

    return 0;

}
