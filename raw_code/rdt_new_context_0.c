rdt_new_context (void)

{

    PayloadContext *rdt = av_mallocz(sizeof(PayloadContext));



    int ret = avformat_open_input(&rdt->rmctx, "", &ff_rdt_demuxer, NULL);

    if (ret < 0) {

        av_free(rdt);

        return NULL;

    }



    return rdt;

}
