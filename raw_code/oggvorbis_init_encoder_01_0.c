static av_cold int oggvorbis_init_encoder(vorbis_info *vi, AVCodecContext *avccontext)

{

    OggVorbisContext *context = avccontext->priv_data;

    double cfreq;



    if (avccontext->flags & CODEC_FLAG_QSCALE) {

        /* variable bitrate */

        if (vorbis_encode_setup_vbr(vi, avccontext->channels,

                                    avccontext->sample_rate,

                                    avccontext->global_quality / (float)FF_QP2LAMBDA / 10.0))

            return -1;

    } else {

        int minrate = avccontext->rc_min_rate > 0 ? avccontext->rc_min_rate : -1;

        int maxrate = avccontext->rc_min_rate > 0 ? avccontext->rc_max_rate : -1;



        /* constant bitrate */

        if (vorbis_encode_setup_managed(vi, avccontext->channels,

                                        avccontext->sample_rate, minrate,

                                        avccontext->bit_rate, maxrate))

            return -1;



        /* variable bitrate by estimate, disable slow rate management */

        if (minrate == -1 && maxrate == -1)

            if (vorbis_encode_ctl(vi, OV_ECTL_RATEMANAGE2_SET, NULL))

                return -1;

    }



    /* cutoff frequency */

    if (avccontext->cutoff > 0) {

        cfreq = avccontext->cutoff / 1000.0;

        if (vorbis_encode_ctl(vi, OV_ECTL_LOWPASS_SET, &cfreq))

            return -1;

    }



    if (context->iblock) {

        vorbis_encode_ctl(vi, OV_ECTL_IBLOCK_SET, &context->iblock);

    }



    return vorbis_encode_setup_init(vi);

}
