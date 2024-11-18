static av_cold int Faac_encode_close(AVCodecContext *avctx)

{

    FaacAudioContext *s = avctx->priv_data;



    av_freep(&avctx->coded_frame);

    av_freep(&avctx->extradata);



    faacEncClose(s->faac_handle);

    return 0;

}
