av_cold int vp56_free(AVCodecContext *avctx)
{
    VP56Context *s = avctx->priv_data;
    int pt;
    av_freep(&s->qscale_table);
    av_freep(&s->above_blocks);
    av_freep(&s->macroblocks);
    av_freep(&s->edge_emu_buffer_alloc);
    if (s->framep[VP56_FRAME_GOLDEN]->data[0])
        avctx->release_buffer(avctx, s->framep[VP56_FRAME_GOLDEN]);
    if (s->framep[VP56_FRAME_GOLDEN2]->data[0])
        avctx->release_buffer(avctx, s->framep[VP56_FRAME_GOLDEN2]);
    if (s->framep[VP56_FRAME_PREVIOUS]->data[0])
        avctx->release_buffer(avctx, s->framep[VP56_FRAME_PREVIOUS]);
    return 0;