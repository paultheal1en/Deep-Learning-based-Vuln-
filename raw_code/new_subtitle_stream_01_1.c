static void new_subtitle_stream(AVFormatContext *oc, int file_idx)

{

    AVStream *st;

    AVOutputStream *ost;

    AVCodec *codec=NULL;

    AVCodecContext *subtitle_enc;

    enum CodecID codec_id;



    st = av_new_stream(oc, oc->nb_streams < nb_streamid_map ? streamid_map[oc->nb_streams] : 0);

    if (!st) {

        fprintf(stderr, "Could not alloc stream\n");

        ffmpeg_exit(1);

    }

    ost = new_output_stream(oc, file_idx);

    subtitle_enc = st->codec;

    output_codecs = grow_array(output_codecs, sizeof(*output_codecs), &nb_output_codecs, nb_output_codecs + 1);

    if(!subtitle_stream_copy){

        if (subtitle_codec_name) {

            codec_id = find_codec_or_die(subtitle_codec_name, AVMEDIA_TYPE_SUBTITLE, 1,

                                         avcodec_opts[AVMEDIA_TYPE_SUBTITLE]->strict_std_compliance);

            codec= output_codecs[nb_output_codecs-1] = avcodec_find_encoder_by_name(subtitle_codec_name);

        } else {

            codec_id = av_guess_codec(oc->oformat, NULL, oc->filename, NULL, AVMEDIA_TYPE_SUBTITLE);

            codec = avcodec_find_encoder(codec_id);

        }

    }

    avcodec_get_context_defaults3(st->codec, codec);



    ost->bitstream_filters = subtitle_bitstream_filters;

    subtitle_bitstream_filters= NULL;



    subtitle_enc->codec_type = AVMEDIA_TYPE_SUBTITLE;



    if(subtitle_codec_tag)

        subtitle_enc->codec_tag= subtitle_codec_tag;



    if (oc->oformat->flags & AVFMT_GLOBALHEADER) {

        subtitle_enc->flags |= CODEC_FLAG_GLOBAL_HEADER;

        avcodec_opts[AVMEDIA_TYPE_SUBTITLE]->flags |= CODEC_FLAG_GLOBAL_HEADER;

    }

    if (subtitle_stream_copy) {

        st->stream_copy = 1;

    } else {

        subtitle_enc->codec_id = codec_id;

        set_context_opts(avcodec_opts[AVMEDIA_TYPE_SUBTITLE], subtitle_enc, AV_OPT_FLAG_SUBTITLE_PARAM | AV_OPT_FLAG_ENCODING_PARAM, codec);

    }



    if (subtitle_language) {

        av_metadata_set2(&st->metadata, "language", subtitle_language, 0);

        av_freep(&subtitle_language);

    }



    subtitle_disable = 0;

    av_freep(&subtitle_codec_name);

    subtitle_stream_copy = 0;

}
