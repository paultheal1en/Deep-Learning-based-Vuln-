static int libquvi_close(AVFormatContext *s)

{

    LibQuviContext *qc = s->priv_data;

    if (qc->fmtctx)

        avformat_close_input(&qc->fmtctx);

    return 0;

}
