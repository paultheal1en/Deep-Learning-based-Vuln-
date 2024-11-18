static int flv_same_audio_codec(AVCodecContext *acodec, int flags)

{

    int bits_per_coded_sample = (flags & FLV_AUDIO_SAMPLESIZE_MASK) ? 16 : 8;

    int flv_codecid = flags & FLV_AUDIO_CODECID_MASK;

    int codec_id;



    if (!acodec->codec_id && !acodec->codec_tag)

        return 1;



    if (acodec->bits_per_coded_sample != bits_per_coded_sample)

        return 0;



    switch(flv_codecid) {

        //no distinction between S16 and S8 PCM codec flags

    case FLV_CODECID_PCM:

        codec_id = bits_per_coded_sample == 8 ? AV_CODEC_ID_PCM_U8 :

#if HAVE_BIGENDIAN

                            AV_CODEC_ID_PCM_S16BE;

#else

                            AV_CODEC_ID_PCM_S16LE;

#endif

        return codec_id == acodec->codec_id;

    case FLV_CODECID_PCM_LE:

        codec_id = bits_per_coded_sample == 8 ? AV_CODEC_ID_PCM_U8 : AV_CODEC_ID_PCM_S16LE;

        return codec_id == acodec->codec_id;

    case FLV_CODECID_AAC:

        return acodec->codec_id == AV_CODEC_ID_AAC;

    case FLV_CODECID_ADPCM:

        return acodec->codec_id == AV_CODEC_ID_ADPCM_SWF;

    case FLV_CODECID_SPEEX:

        return acodec->codec_id == AV_CODEC_ID_SPEEX;

    case FLV_CODECID_MP3:

        return acodec->codec_id == AV_CODEC_ID_MP3;

    case FLV_CODECID_NELLYMOSER_8KHZ_MONO:

    case FLV_CODECID_NELLYMOSER_16KHZ_MONO:

    case FLV_CODECID_NELLYMOSER:

        return acodec->codec_id == AV_CODEC_ID_NELLYMOSER;

    case FLV_CODECID_PCM_MULAW:

        return acodec->sample_rate == 8000 &&

               acodec->codec_id == AV_CODEC_ID_PCM_MULAW;

    case FLV_CODECID_PCM_ALAW:

        return acodec->sample_rate = 8000 &&

               acodec->codec_id == AV_CODEC_ID_PCM_ALAW;

    default:

        return acodec->codec_tag == (flv_codecid >> FLV_AUDIO_CODECID_OFFSET);

    }

}
