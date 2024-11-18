static av_cold int vqa_decode_end(AVCodecContext *avctx)

{

    VqaContext *s = avctx->priv_data;



    av_free(s->codebook);

    av_free(s->next_codebook_buffer);

    av_free(s->decode_buffer);



    if (s->frame.data[0])

        avctx->release_buffer(avctx, &s->frame);



    return 0;

}
