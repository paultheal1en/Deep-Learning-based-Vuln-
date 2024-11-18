static av_cold int adpcm_encode_init(AVCodecContext *avctx)

{

    ADPCMEncodeContext *s = avctx->priv_data;

    uint8_t *extradata;

    int i;

    if (avctx->channels > 2)

        return -1; /* only stereo or mono =) */



    if (avctx->trellis && (unsigned)avctx->trellis > 16U) {

        av_log(avctx, AV_LOG_ERROR, "invalid trellis size\n");

        return -1;

    }



    if (avctx->trellis) {

        int frontier  = 1 << avctx->trellis;

        int max_paths =  frontier * FREEZE_INTERVAL;

        FF_ALLOC_OR_GOTO(avctx, s->paths,

                         max_paths * sizeof(*s->paths), error);

        FF_ALLOC_OR_GOTO(avctx, s->node_buf,

                         2 * frontier * sizeof(*s->node_buf),  error);

        FF_ALLOC_OR_GOTO(avctx, s->nodep_buf,

                         2 * frontier * sizeof(*s->nodep_buf), error);

        FF_ALLOC_OR_GOTO(avctx, s->trellis_hash,

                         65536 * sizeof(*s->trellis_hash), error);

    }



    avctx->bits_per_coded_sample = av_get_bits_per_sample(avctx->codec->id);



    switch (avctx->codec->id) {

    case CODEC_ID_ADPCM_IMA_WAV:

        /* each 16 bits sample gives one nibble

           and we have 4 bytes per channel overhead */

        avctx->frame_size = (BLKSIZE - 4 * avctx->channels) * 8 /

                            (4 * avctx->channels) + 1;

        /* seems frame_size isn't taken into account...

           have to buffer the samples :-( */

        avctx->block_align = BLKSIZE;

        break;

    case CODEC_ID_ADPCM_IMA_QT:

        avctx->frame_size  = 64;

        avctx->block_align = 34 * avctx->channels;

        break;

    case CODEC_ID_ADPCM_MS:

        /* each 16 bits sample gives one nibble

           and we have 7 bytes per channel overhead */

        avctx->frame_size = (BLKSIZE - 7 * avctx->channels) * 2 /

                             avctx->channels + 2;

        avctx->block_align    = BLKSIZE;

        avctx->extradata_size = 32;

        extradata = avctx->extradata = av_malloc(avctx->extradata_size);

        if (!extradata)

            return AVERROR(ENOMEM);

        bytestream_put_le16(&extradata, avctx->frame_size);

        bytestream_put_le16(&extradata, 7); /* wNumCoef */

        for (i = 0; i < 7; i++) {

            bytestream_put_le16(&extradata, ff_adpcm_AdaptCoeff1[i] * 4);

            bytestream_put_le16(&extradata, ff_adpcm_AdaptCoeff2[i] * 4);

        }

        break;

    case CODEC_ID_ADPCM_YAMAHA:

        avctx->frame_size  = BLKSIZE * avctx->channels;

        avctx->block_align = BLKSIZE;

        break;

    case CODEC_ID_ADPCM_SWF:

        if (avctx->sample_rate != 11025 &&

            avctx->sample_rate != 22050 &&

            avctx->sample_rate != 44100) {

            av_log(avctx, AV_LOG_ERROR, "Sample rate must be 11025, "

                   "22050 or 44100\n");

            goto error;

        }

        avctx->frame_size = 512 * (avctx->sample_rate / 11025);

        break;

    default:

        goto error;

    }



    avctx->coded_frame = avcodec_alloc_frame();



    return 0;

error:

    av_freep(&s->paths);

    av_freep(&s->node_buf);

    av_freep(&s->nodep_buf);

    av_freep(&s->trellis_hash);

    return -1;

}
