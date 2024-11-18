static av_cold int twin_decode_close(AVCodecContext *avctx)

{

    TwinContext *tctx = avctx->priv_data;

    int i;



    for (i = 0; i < 3; i++) {

        ff_mdct_end(&tctx->mdct_ctx[i]);

        av_free(tctx->cos_tabs[i]);

    }





    av_free(tctx->curr_frame);

    av_free(tctx->spectrum);

    av_free(tctx->prev_frame);

    av_free(tctx->tmp_buf);



    return 0;

}
