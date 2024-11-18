static av_cold int flic_decode_init(AVCodecContext *avctx)

{

    FlicDecodeContext *s = avctx->priv_data;

    unsigned char *fli_header = (unsigned char *)avctx->extradata;

    int depth;



    if (avctx->extradata_size != 0 &&

        avctx->extradata_size != 12 &&

        avctx->extradata_size != 128 &&

        avctx->extradata_size != 1024) {

        av_log(avctx, AV_LOG_ERROR, "Expected extradata of 12, 128 or 1024 bytes\n");

        return AVERROR_INVALIDDATA;

    }



    s->avctx = avctx;



    if (s->avctx->extradata_size == 12) {

        /* special case for magic carpet FLIs */

        s->fli_type = FLC_MAGIC_CARPET_SYNTHETIC_TYPE_CODE;

        depth = 8;

    } else if (avctx->extradata_size == 1024) {

        uint8_t *ptr = avctx->extradata;

        int i;



        for (i = 0; i < 256; i++) {

            s->palette[i] = AV_RL32(ptr);

            ptr += 4;

        }

        depth = 8;

    } else if (avctx->extradata_size == 0) {

        /* FLI in MOV, see e.g. FFmpeg trac issue #626 */

        s->fli_type = FLI_TYPE_CODE;

        depth = 8;

    } else {

        s->fli_type = AV_RL16(&fli_header[4]);

        depth = AV_RL16(&fli_header[12]);

    }



    if (depth == 0) {

        depth = 8; /* Some FLC generators set depth to zero, when they mean 8Bpp. Fix up here */

    }



    if ((s->fli_type == FLC_FLX_TYPE_CODE) && (depth == 16)) {

        depth = 15; /* Original Autodesk FLX's say the depth is 16Bpp when it is really 15Bpp */

    }



    switch (depth) {

        case 8  : avctx->pix_fmt = PIX_FMT_PAL8; break;

        case 15 : avctx->pix_fmt = PIX_FMT_RGB555; break;

        case 16 : avctx->pix_fmt = PIX_FMT_RGB565; break;

        case 24 : avctx->pix_fmt = PIX_FMT_BGR24; /* Supposedly BGR, but havent any files to test with */

                  av_log(avctx, AV_LOG_ERROR, "24Bpp FLC/FLX is unsupported due to no test files.\n");

                  return -1;

        default :

                  av_log(avctx, AV_LOG_ERROR, "Unknown FLC/FLX depth of %d Bpp is unsupported.\n",depth);

                  return -1;

    }



    avcodec_get_frame_defaults(&s->frame);

    s->frame.data[0] = NULL;

    s->new_palette = 0;



    return 0;

}
