static int init_image(TiffContext *s, ThreadFrame *frame)

{

    int ret;



    switch (s->planar * 1000 + s->bpp * 10 + s->bppcount) {

    case 11:

        if (!s->palette_is_set) {

            s->avctx->pix_fmt = AV_PIX_FMT_MONOBLACK;

            break;

        }

    case 21:

    case 41:

    case 81:

        s->avctx->pix_fmt = s->palette_is_set ? AV_PIX_FMT_PAL8 : AV_PIX_FMT_GRAY8;

        break;

    case 243:

        if (s->photometric == TIFF_PHOTOMETRIC_YCBCR) {

            if (s->subsampling[0] == 1 && s->subsampling[1] == 1) {

                s->avctx->pix_fmt = AV_PIX_FMT_YUV444P;

            } else if (s->subsampling[0] == 2 && s->subsampling[1] == 1) {

                s->avctx->pix_fmt = AV_PIX_FMT_YUV422P;

            } else if (s->subsampling[0] == 4 && s->subsampling[1] == 1) {

                s->avctx->pix_fmt = AV_PIX_FMT_YUV411P;

            } else if (s->subsampling[0] == 1 && s->subsampling[1] == 2) {

                s->avctx->pix_fmt = AV_PIX_FMT_YUV440P;

            } else if (s->subsampling[0] == 2 && s->subsampling[1] == 2) {

                s->avctx->pix_fmt = AV_PIX_FMT_YUV420P;

            } else if (s->subsampling[0] == 4 && s->subsampling[1] == 4) {

                s->avctx->pix_fmt = AV_PIX_FMT_YUV410P;

            } else {

                av_log(s->avctx, AV_LOG_ERROR, "Unsupported YCbCr subsampling\n");

                return AVERROR_PATCHWELCOME;

            }

        } else

            s->avctx->pix_fmt = AV_PIX_FMT_RGB24;

        break;

    case 161:

        s->avctx->pix_fmt = s->le ? AV_PIX_FMT_GRAY16LE : AV_PIX_FMT_GRAY16BE;

        break;

    case 162:

        s->avctx->pix_fmt = AV_PIX_FMT_YA8;

        break;

    case 322:

        s->avctx->pix_fmt = s->le ? AV_PIX_FMT_YA16LE : AV_PIX_FMT_YA16BE;

        break;

    case 324:

        s->avctx->pix_fmt = AV_PIX_FMT_RGBA;

        break;

    case 483:

        s->avctx->pix_fmt = s->le ? AV_PIX_FMT_RGB48LE  : AV_PIX_FMT_RGB48BE;

        break;

    case 644:

        s->avctx->pix_fmt = s->le ? AV_PIX_FMT_RGBA64LE  : AV_PIX_FMT_RGBA64BE;

        break;

    case 1243:

        s->avctx->pix_fmt = AV_PIX_FMT_GBRP;

        break;

    case 1324:

        s->avctx->pix_fmt = AV_PIX_FMT_GBRAP;

        break;

    case 1483:

        s->avctx->pix_fmt = s->le ? AV_PIX_FMT_GBRP16LE : AV_PIX_FMT_GBRP16BE;

        break;

    case 1644:

        s->avctx->pix_fmt = s->le ? AV_PIX_FMT_GBRAP16LE : AV_PIX_FMT_GBRAP16BE;

        break;

    default:

        av_log(s->avctx, AV_LOG_ERROR,

               "This format is not supported (bpp=%d, bppcount=%d)\n",

               s->bpp, s->bppcount);

        return AVERROR_INVALIDDATA;

    }



    if (s->photometric == TIFF_PHOTOMETRIC_YCBCR) {

        const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(s->avctx->pix_fmt);

        if((desc->flags & AV_PIX_FMT_FLAG_RGB) || desc->nb_components < 3) {

            av_log(s->avctx, AV_LOG_ERROR, "Unsupported YCbCr variant\n");

            return AVERROR_INVALIDDATA;

        }

    }



    if (s->width != s->avctx->width || s->height != s->avctx->height) {

        ret = ff_set_dimensions(s->avctx, s->width, s->height);

        if (ret < 0)

            return ret;

    }

    if ((ret = ff_thread_get_buffer(s->avctx, frame, 0)) < 0)

        return ret;

    if (s->avctx->pix_fmt == AV_PIX_FMT_PAL8) {

        memcpy(frame->f->data[1], s->palette, sizeof(s->palette));

    }

    return 0;

}
