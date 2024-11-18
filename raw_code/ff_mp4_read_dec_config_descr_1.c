int ff_mp4_read_dec_config_descr(AVFormatContext *fc, AVStream *st, AVIOContext *pb)

{

    int len, tag;

    int object_type_id = avio_r8(pb);

    avio_r8(pb); /* stream type */

    avio_rb24(pb); /* buffer size db */

    avio_rb32(pb); /* max bitrate */

    avio_rb32(pb); /* avg bitrate */



    if(avcodec_is_open(st->codec)) {

        av_log(fc, AV_LOG_DEBUG, "codec open in read_dec_config_descr\n");

        return -1;

    }



    st->codec->codec_id= ff_codec_get_id(ff_mp4_obj_type, object_type_id);

    av_dlog(fc, "esds object type id 0x%02x\n", object_type_id);

    len = ff_mp4_read_descr(fc, pb, &tag);

    if (tag == MP4DecSpecificDescrTag) {

        av_dlog(fc, "Specific MPEG4 header len=%d\n", len);

        if (!len || (uint64_t)len > (1<<30))

            return -1;

        av_free(st->codec->extradata);

        if (ff_alloc_extradata(st->codec, len))

            return AVERROR(ENOMEM);

        avio_read(pb, st->codec->extradata, len);

        if (st->codec->codec_id == AV_CODEC_ID_AAC) {

            MPEG4AudioConfig cfg = {0};

            avpriv_mpeg4audio_get_config(&cfg, st->codec->extradata,

                                         st->codec->extradata_size * 8, 1);

            st->codec->channels = cfg.channels;

            if (cfg.object_type == 29 && cfg.sampling_index < 3) // old mp3on4

                st->codec->sample_rate = avpriv_mpa_freq_tab[cfg.sampling_index];

            else if (cfg.ext_sample_rate)

                st->codec->sample_rate = cfg.ext_sample_rate;

            else

                st->codec->sample_rate = cfg.sample_rate;

            av_dlog(fc, "mp4a config channels %d obj %d ext obj %d "

                    "sample rate %d ext sample rate %d\n", st->codec->channels,

                    cfg.object_type, cfg.ext_object_type,

                    cfg.sample_rate, cfg.ext_sample_rate);

            if (!(st->codec->codec_id = ff_codec_get_id(mp4_audio_types,

                                                        cfg.object_type)))

                st->codec->codec_id = AV_CODEC_ID_AAC;

        }

    }

    return 0;

}
