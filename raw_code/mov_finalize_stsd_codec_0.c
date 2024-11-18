static int mov_finalize_stsd_codec(MOVContext *c, AVIOContext *pb,

                                   AVStream *st, MOVStreamContext *sc)

{

    if (st->codec->codec_type == AVMEDIA_TYPE_AUDIO &&

        !st->codec->sample_rate && sc->time_scale > 1)

        st->codec->sample_rate = sc->time_scale;



    /* special codec parameters handling */

    switch (st->codec->codec_id) {

#if CONFIG_DV_DEMUXER

    case AV_CODEC_ID_DVAUDIO:

        c->dv_fctx = avformat_alloc_context();

        if (!c->dv_fctx) {

            av_log(c->fc, AV_LOG_ERROR, "dv demux context alloc error\n");

            return AVERROR(ENOMEM);

        }

        c->dv_demux = avpriv_dv_init_demux(c->dv_fctx);

        if (!c->dv_demux) {

            av_log(c->fc, AV_LOG_ERROR, "dv demux context init error\n");

            return AVERROR(ENOMEM);

        }

        sc->dv_audio_container = 1;

        st->codec->codec_id    = AV_CODEC_ID_PCM_S16LE;

        break;

#endif

    /* no ifdef since parameters are always those */

    case AV_CODEC_ID_QCELP:

        st->codec->channels = 1;

        // force sample rate for qcelp when not stored in mov

        if (st->codec->codec_tag != MKTAG('Q','c','l','p'))

            st->codec->sample_rate = 8000;

        break;

    case AV_CODEC_ID_AMR_NB:

        st->codec->channels    = 1;

        /* force sample rate for amr, stsd in 3gp does not store sample rate */

        st->codec->sample_rate = 8000;

        break;

    case AV_CODEC_ID_AMR_WB:

        st->codec->channels    = 1;

        st->codec->sample_rate = 16000;

        break;

    case AV_CODEC_ID_MP2:

    case AV_CODEC_ID_MP3:

        /* force type after stsd for m1a hdlr */

        st->codec->codec_type = AVMEDIA_TYPE_AUDIO;

        st->need_parsing      = AVSTREAM_PARSE_FULL;

        break;

    case AV_CODEC_ID_GSM:

    case AV_CODEC_ID_ADPCM_MS:

    case AV_CODEC_ID_ADPCM_IMA_WAV:

    case AV_CODEC_ID_ILBC:

        st->codec->block_align = sc->bytes_per_frame;

        break;

    case AV_CODEC_ID_ALAC:

        if (st->codec->extradata_size == 36) {

            st->codec->channels    = AV_RB8 (st->codec->extradata + 21);

            st->codec->sample_rate = AV_RB32(st->codec->extradata + 32);

        }

        break;

    case AV_CODEC_ID_VC1:

        st->need_parsing = AVSTREAM_PARSE_FULL;

        break;

    default:

        break;

    }

    return 0;

}
