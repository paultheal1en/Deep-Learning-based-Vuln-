static void new_audio_stream(AVFormatContext *oc, int file_idx)

{

    AVStream *st;

    OutputStream *ost;

    AVCodec *codec= NULL;

    AVCodecContext *audio_enc;

    enum CodecID codec_id = CODEC_ID_NONE;



    if(!audio_stream_copy){

        if (audio_codec_name) {

            codec_id = find_codec_or_die(audio_codec_name, AVMEDIA_TYPE_AUDIO, 1,

                                         avcodec_opts[AVMEDIA_TYPE_AUDIO]->strict_std_compliance);

            codec = avcodec_find_encoder_by_name(audio_codec_name);

        } else {

            codec_id = av_guess_codec(oc->oformat, NULL, oc->filename, NULL, AVMEDIA_TYPE_AUDIO);

            codec = avcodec_find_encoder(codec_id);

        }

    }

    ost = new_output_stream(oc, file_idx, codec);

    st  = ost->st;



    ost->bitstream_filters = audio_bitstream_filters;

    audio_bitstream_filters= NULL;



    st->codec->thread_count= thread_count;



    audio_enc = st->codec;

    audio_enc->codec_type = AVMEDIA_TYPE_AUDIO;



    if(audio_codec_tag)

        audio_enc->codec_tag= audio_codec_tag;



    if (oc->oformat->flags & AVFMT_GLOBALHEADER) {

        audio_enc->flags |= CODEC_FLAG_GLOBAL_HEADER;

    }

    if (audio_stream_copy) {

        st->stream_copy = 1;

    } else {

        audio_enc->codec_id = codec_id;

        set_context_opts(audio_enc, avcodec_opts[AVMEDIA_TYPE_AUDIO], AV_OPT_FLAG_AUDIO_PARAM | AV_OPT_FLAG_ENCODING_PARAM, codec);



        if (audio_qscale > QSCALE_NONE) {

            audio_enc->flags |= CODEC_FLAG_QSCALE;

            audio_enc->global_quality = FF_QP2LAMBDA * audio_qscale;

        }

        if (audio_channels)

            audio_enc->channels = audio_channels;

        if (audio_sample_fmt != AV_SAMPLE_FMT_NONE)

            audio_enc->sample_fmt = audio_sample_fmt;

        if (audio_sample_rate)

            audio_enc->sample_rate = audio_sample_rate;

    }

    if (audio_language) {

        av_dict_set(&st->metadata, "language", audio_language, 0);

        av_freep(&audio_language);

    }



    /* reset some key parameters */

    audio_disable = 0;

    av_freep(&audio_codec_name);

    audio_stream_copy = 0;

}
