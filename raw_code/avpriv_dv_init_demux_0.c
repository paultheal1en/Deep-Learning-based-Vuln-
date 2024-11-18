DVDemuxContext* avpriv_dv_init_demux(AVFormatContext *s)

{

    DVDemuxContext *c;



    c = av_mallocz(sizeof(DVDemuxContext));

    if (!c)

        return NULL;



    c->vst = avformat_new_stream(s, NULL);

    if (!c->vst) {

        av_free(c);

        return NULL;

    }



    c->sys  = NULL;

    c->fctx = s;

    memset(c->ast, 0, sizeof(c->ast));

    c->ach    = 0;

    c->frames = 0;

    c->abytes = 0;



    c->vst->codec->codec_type = AVMEDIA_TYPE_VIDEO;

    c->vst->codec->codec_id   = CODEC_ID_DVVIDEO;

    c->vst->codec->bit_rate   = 25000000;

    c->vst->start_time        = 0;



    return c;

}
