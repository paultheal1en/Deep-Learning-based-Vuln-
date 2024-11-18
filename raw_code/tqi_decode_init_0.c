static av_cold int tqi_decode_init(AVCodecContext *avctx)

{

    TqiContext *t = avctx->priv_data;



    ff_blockdsp_init(&t->bdsp, avctx);

    ff_bswapdsp_init(&t->bsdsp);

    ff_idctdsp_init(&t->idsp, avctx);

    ff_init_scantable_permutation(t->idsp.idct_permutation, FF_IDCT_PERM_NONE);

    ff_init_scantable(t->idsp.idct_permutation, &t->intra_scantable, ff_zigzag_direct);



    avctx->framerate = (AVRational){ 15, 1 };

    avctx->pix_fmt = AV_PIX_FMT_YUV420P;

    ff_mpeg12_init_vlcs();

    return 0;

}
