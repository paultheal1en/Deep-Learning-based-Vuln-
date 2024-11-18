av_cold int ff_rv34_decode_init(AVCodecContext *avctx)

{

    RV34DecContext *r = avctx->priv_data;

    MpegEncContext *s = &r->s;



    MPV_decode_defaults(s);

    s->avctx= avctx;

    s->out_format = FMT_H263;

    s->codec_id= avctx->codec_id;



    s->width = avctx->width;

    s->height = avctx->height;



    r->s.avctx = avctx;

    avctx->flags |= CODEC_FLAG_EMU_EDGE;

    r->s.flags |= CODEC_FLAG_EMU_EDGE;

    avctx->pix_fmt = PIX_FMT_YUV420P;

    avctx->has_b_frames = 1;

    s->low_delay = 0;



    if (MPV_common_init(s) < 0)

        return -1;



    ff_h264_pred_init(&r->h, CODEC_ID_RV40);



    r->intra_types_hist = av_malloc(s->b4_stride * 4 * 2 * sizeof(*r->intra_types_hist));

    r->intra_types = r->intra_types_hist + s->b4_stride * 4;



    r->mb_type = av_mallocz(r->s.mb_stride * r->s.mb_height * sizeof(*r->mb_type));



    r->cbp_luma   = av_malloc(r->s.mb_stride * r->s.mb_height * sizeof(*r->cbp_luma));

    r->cbp_chroma = av_malloc(r->s.mb_stride * r->s.mb_height * sizeof(*r->cbp_chroma));

    r->deblock_coefs = av_malloc(r->s.mb_stride * r->s.mb_height * sizeof(*r->deblock_coefs));



    if(!intra_vlcs[0].cbppattern[0].bits)

        rv34_init_tables();



    return 0;

}
