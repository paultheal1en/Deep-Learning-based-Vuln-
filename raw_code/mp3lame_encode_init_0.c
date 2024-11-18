static av_cold int mp3lame_encode_init(AVCodecContext *avctx)

{

    LAMEContext *s = avctx->priv_data;

    int ret;



    s->avctx = avctx;



    /* initialize LAME and get defaults */

    if (!(s->gfp = lame_init()))

        return AVERROR(ENOMEM);



    lame_set_num_channels(s->gfp, avctx->channels);

    lame_set_mode(s->gfp, avctx->channels > 1 ? s->joint_stereo ? JOINT_STEREO : STEREO : MONO);



    /* sample rate */

    lame_set_in_samplerate (s->gfp, avctx->sample_rate);

    lame_set_out_samplerate(s->gfp, avctx->sample_rate);



    /* algorithmic quality */

    if (avctx->compression_level == FF_COMPRESSION_DEFAULT)

        lame_set_quality(s->gfp, 5);

    else

        lame_set_quality(s->gfp, avctx->compression_level);



    /* rate control */

    if (avctx->flags & CODEC_FLAG_QSCALE) { // VBR

        lame_set_VBR(s->gfp, vbr_default);

        lame_set_VBR_quality(s->gfp, avctx->global_quality / (float)FF_QP2LAMBDA);

    } else {

        if (avctx->bit_rate) {

            if (s->abr) {                   // ABR

                lame_set_VBR(s->gfp, vbr_abr);

                lame_set_VBR_mean_bitrate_kbps(s->gfp, avctx->bit_rate / 1000);

            } else                          // CBR

                lame_set_brate(s->gfp, avctx->bit_rate / 1000);

        }

    }



    /* do not get a Xing VBR header frame from LAME */

    lame_set_bWriteVbrTag(s->gfp,0);



    /* bit reservoir usage */

    lame_set_disable_reservoir(s->gfp, !s->reservoir);



    /* set specified parameters */

    if (lame_init_params(s->gfp) < 0) {

        ret = -1;

        goto error;

    }



    /* get encoder delay */

    avctx->delay = lame_get_encoder_delay(s->gfp) + 528 + 1;

    ff_af_queue_init(avctx, &s->afq);



    avctx->frame_size  = lame_get_framesize(s->gfp);



    /* allocate float sample buffers */

    if (avctx->sample_fmt == AV_SAMPLE_FMT_FLTP) {

        int ch;

        for (ch = 0; ch < avctx->channels; ch++) {

            s->samples_flt[ch] = av_malloc(avctx->frame_size *

                                           sizeof(*s->samples_flt[ch]));

            if (!s->samples_flt[ch]) {

                ret = AVERROR(ENOMEM);

                goto error;

            }

        }

    }



    ret = realloc_buffer(s);

    if (ret < 0)

        goto error;



    avpriv_float_dsp_init(&s->fdsp, avctx->flags & CODEC_FLAG_BITEXACT);



    return 0;

error:

    mp3lame_encode_close(avctx);

    return ret;

}
