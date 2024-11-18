static int decode_end(AVCodecContext * avctx)

{

    KmvcContext *const c = (KmvcContext *) avctx->priv_data;



    if (c->frm0)

        av_free(c->frm0);

    if (c->frm1)

        av_free(c->frm1);

    if (c->pic.data[0])

        avctx->release_buffer(avctx, &c->pic);



    return 0;

}
