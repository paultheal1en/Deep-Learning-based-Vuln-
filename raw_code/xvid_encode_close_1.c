static av_cold int xvid_encode_close(AVCodecContext *avctx) {

    struct xvid_context *x = avctx->priv_data;



    xvid_encore(x->encoder_handle, XVID_ENC_DESTROY, NULL, NULL);



    if( avctx->extradata != NULL )

        av_free(avctx->extradata);

    if( x->twopassbuffer != NULL ) {

        av_free(x->twopassbuffer);

        av_free(x->old_twopassbuffer);

    }

    if( x->twopassfile != NULL )

        av_free(x->twopassfile);

    if( x->intra_matrix != NULL )

        av_free(x->intra_matrix);

    if( x->inter_matrix != NULL )

        av_free(x->inter_matrix);



    return 0;

}
