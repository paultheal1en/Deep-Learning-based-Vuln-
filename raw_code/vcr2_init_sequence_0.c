static int vcr2_init_sequence(AVCodecContext *avctx)

{

    Mpeg1Context *s1  = avctx->priv_data;

    MpegEncContext *s = &s1->mpeg_enc_ctx;

    int i, v, ret;



    /* start new MPEG-1 context decoding */

    s->out_format = FMT_MPEG1;

    if (s1->mpeg_enc_ctx_allocated) {

        ff_mpv_common_end(s);

    }

    s->width            = avctx->coded_width;

    s->height           = avctx->coded_height;

    avctx->has_b_frames = 0; // true?

    s->low_delay        = 1;



    avctx->pix_fmt = mpeg_get_pixelformat(avctx);



#if FF_API_XVMC

    if ((avctx->pix_fmt == AV_PIX_FMT_XVMC_MPEG2_IDCT || avctx->hwaccel) &&

        avctx->idct_algo == FF_IDCT_AUTO)

#else

    if (avctx->hwaccel && avctx->idct_algo == FF_IDCT_AUTO)

#endif /* FF_API_XVMC */

        avctx->idct_algo = FF_IDCT_SIMPLE;



    ff_mpv_idct_init(s);

    if ((ret = ff_mpv_common_init(s)) < 0)

        return ret;

    s1->mpeg_enc_ctx_allocated = 1;



    for (i = 0; i < 64; i++) {

        int j = s->idsp.idct_permutation[i];

        v = ff_mpeg1_default_intra_matrix[i];

        s->intra_matrix[j]        = v;

        s->chroma_intra_matrix[j] = v;



        v = ff_mpeg1_default_non_intra_matrix[i];

        s->inter_matrix[j]        = v;

        s->chroma_inter_matrix[j] = v;

    }



    s->progressive_sequence  = 1;

    s->progressive_frame     = 1;

    s->picture_structure     = PICT_FRAME;

    s->frame_pred_frame_dct  = 1;

    s->chroma_format         = 1;

    s->codec_id              = s->avctx->codec_id = AV_CODEC_ID_MPEG2VIDEO;

    s1->save_width           = s->width;

    s1->save_height          = s->height;

    s1->save_progressive_seq = s->progressive_sequence;

    return 0;

}
