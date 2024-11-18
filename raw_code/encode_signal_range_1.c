static void encode_signal_range(VC2EncContext *s)

{

    int idx;

    AVCodecContext *avctx = s->avctx;

    const AVPixFmtDescriptor *fmt = av_pix_fmt_desc_get(avctx->pix_fmt);

    const int depth = fmt->comp[0].depth;

    if (depth == 8 && avctx->color_range == AVCOL_RANGE_JPEG) {

        idx = 1;

        s->bpp = 1;

        s->diff_offset = 128;

    } else if (depth == 8 && (avctx->color_range == AVCOL_RANGE_MPEG ||

               avctx->color_range == AVCOL_RANGE_UNSPECIFIED)) {

        idx = 2;

        s->bpp = 1;

        s->diff_offset = 128;

    } else if (depth == 10) {

        idx = 3;

        s->bpp = 2;

        s->diff_offset = 512;

    } else {

        idx = 4;

        s->bpp = 2;

        s->diff_offset = 2048;

    }

    put_bits(&s->pb, 1, !s->strict_compliance);

    if (!s->strict_compliance)

        put_vc2_ue_uint(&s->pb, idx);

}
