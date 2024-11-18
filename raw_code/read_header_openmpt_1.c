static int read_header_openmpt(AVFormatContext *s)

{

    AVStream *st;

    OpenMPTContext *openmpt = s->priv_data;

    int64_t size = avio_size(s->pb);

    if (!size)

        return AVERROR_INVALIDDATA;

    char *buf = av_malloc(size);

    int ret;





    if (!buf)

        return AVERROR(ENOMEM);

    size = avio_read(s->pb, buf, size);

    if (size < 0) {

        av_log(s, AV_LOG_ERROR, "Reading input buffer failed.\n");

        av_freep(&buf);

        return size;

    }



    openmpt->module = openmpt_module_create_from_memory(buf, size, openmpt_logfunc, s, NULL);

    av_freep(&buf);

    if (!openmpt->module)

            return AVERROR_INVALIDDATA;



    openmpt->channels   = av_get_channel_layout_nb_channels(openmpt->layout);

    openmpt->duration   = openmpt_module_get_duration_seconds(openmpt->module);



    add_meta(s, "artist",  openmpt_module_get_metadata(openmpt->module, "artist"));

    add_meta(s, "title",   openmpt_module_get_metadata(openmpt->module, "title"));

    add_meta(s, "encoder", openmpt_module_get_metadata(openmpt->module, "tracker"));

    add_meta(s, "comment", openmpt_module_get_metadata(openmpt->module, "message"));

    add_meta(s, "date",    openmpt_module_get_metadata(openmpt->module, "date"));



    if (openmpt->subsong >= openmpt_module_get_num_subsongs(openmpt->module)) {

        openmpt_module_destroy(openmpt->module);

        av_log(s, AV_LOG_ERROR, "Invalid subsong index: %d\n", openmpt->subsong);

        return AVERROR(EINVAL);

    }



    if (openmpt->subsong != -2) {

        if (openmpt->subsong >= 0) {

            av_dict_set_int(&s->metadata, "track", openmpt->subsong + 1, 0);

        }

        ret = openmpt_module_select_subsong(openmpt->module, openmpt->subsong);

        if (!ret){

            openmpt_module_destroy(openmpt->module);

            av_log(s, AV_LOG_ERROR, "Could not select requested subsong: %d", openmpt->subsong);

            return AVERROR(EINVAL);

        }

    }



    st = avformat_new_stream(s, NULL);

    if (!st) {

        openmpt_module_destroy(openmpt->module);

        openmpt->module = NULL;

        return AVERROR(ENOMEM);

    }

    avpriv_set_pts_info(st, 64, 1, AV_TIME_BASE);

    st->duration = llrint(openmpt->duration*AV_TIME_BASE);



    st->codecpar->codec_type  = AVMEDIA_TYPE_AUDIO;

    st->codecpar->codec_id    = AV_NE(AV_CODEC_ID_PCM_F32BE, AV_CODEC_ID_PCM_F32LE);

    st->codecpar->channels    = openmpt->channels;

    st->codecpar->sample_rate = openmpt->sample_rate;



    return 0;

}
