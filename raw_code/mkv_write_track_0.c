static int mkv_write_track(AVFormatContext *s, MatroskaMuxContext *mkv,

                           int i, AVIOContext *pb)

{

    AVStream *st = s->streams[i];

    AVCodecContext *codec = st->codec;

    ebml_master subinfo, track;

    int native_id = 0;

    int qt_id = 0;

    int bit_depth = av_get_bits_per_sample(codec->codec_id);

    int sample_rate = codec->sample_rate;

    int output_sample_rate = 0;

    int j, ret;

    AVDictionaryEntry *tag;



    // ms precision is the de-facto standard timescale for mkv files

    avpriv_set_pts_info(st, 64, 1, 1000);



    if (codec->codec_type == AVMEDIA_TYPE_ATTACHMENT) {

        mkv->have_attachments = 1;

        return 0;

    }



    if (!bit_depth)

        bit_depth = av_get_bytes_per_sample(codec->sample_fmt) << 3;



    if (codec->codec_id == AV_CODEC_ID_AAC)

        get_aac_sample_rates(s, codec, &sample_rate, &output_sample_rate);



    track = start_ebml_master(pb, MATROSKA_ID_TRACKENTRY, 0);

    put_ebml_uint (pb, MATROSKA_ID_TRACKNUMBER     , i + 1);

    put_ebml_uint (pb, MATROSKA_ID_TRACKUID        , i + 1);

    put_ebml_uint (pb, MATROSKA_ID_TRACKFLAGLACING , 0);    // no lacing (yet)



    if ((tag = av_dict_get(st->metadata, "title", NULL, 0)))

        put_ebml_string(pb, MATROSKA_ID_TRACKNAME, tag->value);

    tag = av_dict_get(st->metadata, "language", NULL, 0);

    put_ebml_string(pb, MATROSKA_ID_TRACKLANGUAGE, tag ? tag->value:"und");



    // The default value for TRACKFLAGDEFAULT is 1, so add element

    // if we need to clear it.

    if (!(st->disposition & AV_DISPOSITION_DEFAULT))

        put_ebml_uint(pb, MATROSKA_ID_TRACKFLAGDEFAULT, !!(st->disposition & AV_DISPOSITION_DEFAULT));



    if (codec->codec_type == AVMEDIA_TYPE_AUDIO && codec->initial_padding) {

        mkv->tracks[i].ts_offset = av_rescale_q(codec->initial_padding,

                                                (AVRational){ 1, codec->sample_rate },

                                                st->time_base);



        put_ebml_uint(pb, MATROSKA_ID_CODECDELAY,

                      av_rescale_q(codec->initial_padding,

                                   (AVRational){ 1, codec->sample_rate },

                                   (AVRational){ 1, 1000000000 }));

    }



    // look for a codec ID string specific to mkv to use,

    // if none are found, use AVI codes

    for (j = 0; ff_mkv_codec_tags[j].id != AV_CODEC_ID_NONE; j++) {

        if (ff_mkv_codec_tags[j].id == codec->codec_id) {

            put_ebml_string(pb, MATROSKA_ID_CODECID, ff_mkv_codec_tags[j].str);

            native_id = 1;

            break;

        }

    }



    if (mkv->mode == MODE_WEBM && !(codec->codec_id == AV_CODEC_ID_VP8 ||

                                    codec->codec_id == AV_CODEC_ID_VP9 ||

                                    codec->codec_id == AV_CODEC_ID_OPUS ||

                                    codec->codec_id == AV_CODEC_ID_VORBIS)) {

        av_log(s, AV_LOG_ERROR,

               "Only VP8 or VP9 video and Vorbis or Opus audio are supported for WebM.\n");

        return AVERROR(EINVAL);

    }



    switch (codec->codec_type) {

    case AVMEDIA_TYPE_VIDEO:

        put_ebml_uint(pb, MATROSKA_ID_TRACKTYPE, MATROSKA_TRACK_TYPE_VIDEO);

        if (st->avg_frame_rate.num > 0 && st->avg_frame_rate.den > 0)

            put_ebml_uint(pb, MATROSKA_ID_TRACKDEFAULTDURATION, 1E9 / av_q2d(st->avg_frame_rate));



        if (!native_id &&

            ff_codec_get_tag(ff_codec_movvideo_tags, codec->codec_id) &&

            (!ff_codec_get_tag(ff_codec_bmp_tags,   codec->codec_id) ||

             codec->codec_id == AV_CODEC_ID_SVQ1 ||

             codec->codec_id == AV_CODEC_ID_SVQ3 ||

             codec->codec_id == AV_CODEC_ID_CINEPAK))

            qt_id = 1;



        if (qt_id)

            put_ebml_string(pb, MATROSKA_ID_CODECID, "V_QUICKTIME");

        else if (!native_id) {

            // if there is no mkv-specific codec ID, use VFW mode

            put_ebml_string(pb, MATROSKA_ID_CODECID, "V_MS/VFW/FOURCC");

            mkv->tracks[i].write_dts = 1;

        }



        subinfo = start_ebml_master(pb, MATROSKA_ID_TRACKVIDEO, 0);

        // XXX: interlace flag?

        put_ebml_uint (pb, MATROSKA_ID_VIDEOPIXELWIDTH , codec->width);

        put_ebml_uint (pb, MATROSKA_ID_VIDEOPIXELHEIGHT, codec->height);



        // check both side data and metadata for stereo information,

        // write the result to the bitstream if any is found

        ret = mkv_write_stereo_mode(s, pb, st, mkv->mode);

        if (ret < 0)

            return ret;



        end_ebml_master(pb, subinfo);

        break;



    case AVMEDIA_TYPE_AUDIO:

        put_ebml_uint(pb, MATROSKA_ID_TRACKTYPE, MATROSKA_TRACK_TYPE_AUDIO);



        if (!native_id)

            // no mkv-specific ID, use ACM mode

            put_ebml_string(pb, MATROSKA_ID_CODECID, "A_MS/ACM");



        subinfo = start_ebml_master(pb, MATROSKA_ID_TRACKAUDIO, 0);

        put_ebml_uint  (pb, MATROSKA_ID_AUDIOCHANNELS    , codec->channels);

        put_ebml_float (pb, MATROSKA_ID_AUDIOSAMPLINGFREQ, sample_rate);

        if (output_sample_rate)

            put_ebml_float(pb, MATROSKA_ID_AUDIOOUTSAMPLINGFREQ, output_sample_rate);

        if (bit_depth)

            put_ebml_uint(pb, MATROSKA_ID_AUDIOBITDEPTH, bit_depth);

        end_ebml_master(pb, subinfo);

        break;



    case AVMEDIA_TYPE_SUBTITLE:

        put_ebml_uint(pb, MATROSKA_ID_TRACKTYPE, MATROSKA_TRACK_TYPE_SUBTITLE);

        if (!native_id) {

            av_log(s, AV_LOG_ERROR, "Subtitle codec %d is not supported.\n", codec->codec_id);

            return AVERROR(ENOSYS);

        }

        break;

    default:

        av_log(s, AV_LOG_ERROR, "Only audio, video, and subtitles are supported for Matroska.\n");

        break;

    }

    ret = mkv_write_codecprivate(s, pb, codec, native_id, qt_id);

    if (ret < 0)

        return ret;



    end_ebml_master(pb, track);



    return 0;

}
