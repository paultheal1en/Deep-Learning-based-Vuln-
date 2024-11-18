static int update_dimensions(VP8Context *s, int width, int height)

{

    int i;



    if (avcodec_check_dimensions(s->avctx, width, height))

        return AVERROR_INVALIDDATA;



    vp8_decode_flush(s->avctx);



    avcodec_set_dimensions(s->avctx, width, height);



    s->mb_width  = (s->avctx->coded_width +15) / 16;

    s->mb_height = (s->avctx->coded_height+15) / 16;



    // we allocate a border around the top/left of intra4x4 modes

    // this is 4 blocks for intra4x4 to keep 4-byte alignment for fill_rectangle

    s->mb_stride = s->mb_width+1;

    s->b4_stride = 4*s->mb_stride;



    s->macroblocks_base        = av_mallocz(s->mb_stride*(s->mb_height+1)*sizeof(*s->macroblocks));

    s->intra4x4_pred_mode_base = av_mallocz(s->b4_stride*(4*s->mb_height+1));

    s->top_nnz                 = av_mallocz(s->mb_width*sizeof(*s->top_nnz));



    if (!s->macroblocks_base || !s->intra4x4_pred_mode_base || !s->top_nnz)

        return AVERROR(ENOMEM);



    s->macroblocks        = s->macroblocks_base        + 1 + s->mb_stride;

    s->intra4x4_pred_mode = s->intra4x4_pred_mode_base + 4 + s->b4_stride;



    memset(s->intra4x4_pred_mode_base, DC_PRED, s->b4_stride);

    for (i = 0; i < 4*s->mb_height; i++)

        s->intra4x4_pred_mode[i*s->b4_stride-1] = DC_PRED;



    return 0;

}
