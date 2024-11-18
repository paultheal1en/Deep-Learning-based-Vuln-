static int vp3_decode_end(AVCodecContext *avctx)

{

    Vp3DecodeContext *s = avctx->priv_data;



    av_free(s->all_fragments);

    av_free(s->coded_fragment_list);

    av_free(s->superblock_fragments);

    av_free(s->superblock_macroblocks);

    av_free(s->macroblock_fragments);

    av_free(s->macroblock_coded);



    /* release all frames */

    avctx->release_buffer(avctx, &s->golden_frame);

    avctx->release_buffer(avctx, &s->last_frame);

    avctx->release_buffer(avctx, &s->current_frame);



    return 0;

}
