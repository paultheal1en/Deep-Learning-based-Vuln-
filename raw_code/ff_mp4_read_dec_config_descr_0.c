int ff_mp4_read_dec_config_descr(AVFormatContext *fc, AVStream *st, AVIOContext *pb)

{

    enum AVCodecID codec_id;

    unsigned v;

    int len, tag;

    int ret;

    int object_type_id = avio_r8(pb);

    avio_r8(pb); /* stream type */

    avio_rb24(pb); /* buffer size db */



    v = avio_rb32(pb);



    // TODO: fix this with codecpar

#if FF_API_LAVF_AVCTX

FF_DISABLE_DEPRECATION_WARNINGS

    if (v < INT32_MAX)

        st->codec->rc_max_rate = v;

FF_ENABLE_DEPRECATION_WARNINGS

#endif



    st->codecpar->bit_rate = avio_rb32(pb); /* avg bitrate */



    codec_id= ff_codec_get_id(ff_mp4_obj_type, object_type_id);

    if (codec_id)

        st->codecpar->codec_id = codec_id;

    av_log(fc, AV_LOG_TRACE, "esds object type id 0x%02x\n", object_type_id);

    len = ff_mp4_read_descr(fc, pb, &tag);

    if (tag == MP4DecSpecificDescrTag) {

        av_log(fc, AV_LOG_TRACE, "Specific MPEG-4 header len=%d\n", len);

        if (!len || (uint64_t)len > (1<<30))

            return -1;

        av_free(st->codecpar->extradata);

        if ((ret = ff_get_extradata(fc, st->codecpar, pb, len)) < 0)

            return ret;

        if (st->codecpar->codec_id == AV_CODEC_ID_AAC) {

            MPEG4AudioConfig cfg = {0};

            avpriv_mpeg4audio_get_config(&cfg, st->codecpar->extradata,

                                         st->codecpar->extradata_size * 8, 1);

            st->codecpar->channels = cfg.channels;

            if (cfg.object_type == 29 && cfg.sampling_index < 3) // old mp3on4

                st->codecpar->sample_rate = avpriv_mpa_freq_tab[cfg.sampling_index];

            else if (cfg.ext_sample_rate)

                st->codecpar->sample_rate = cfg.ext_sample_rate;

            else

                st->codecpar->sample_rate = cfg.sample_rate;

            av_log(fc, AV_LOG_TRACE, "mp4a config channels %d obj %d ext obj %d "

                    "sample rate %d ext sample rate %d\n", st->codecpar->channels,

                    cfg.object_type, cfg.ext_object_type,

                    cfg.sample_rate, cfg.ext_sample_rate);

            if (!(st->codecpar->codec_id = ff_codec_get_id(mp4_audio_types,

                                                        cfg.object_type)))

                st->codecpar->codec_id = AV_CODEC_ID_AAC;

        }

    }

    return 0;

}
