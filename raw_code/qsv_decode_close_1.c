static av_cold int qsv_decode_close(AVCodecContext *avctx)
{
    QSVOtherContext *s = avctx->priv_data;
    ff_qsv_decode_close(&s->qsv);
    qsv_clear_buffers(s);
    av_fifo_free(s->packet_fifo);
    return 0;
}