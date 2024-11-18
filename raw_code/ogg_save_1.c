static int ogg_save(AVFormatContext *s)

{

    struct ogg *ogg = s->priv_data;

    struct ogg_state *ost =

        av_malloc(sizeof (*ost) + (ogg->nstreams-1) * sizeof (*ogg->streams));

    int i;

    ost->pos = avio_tell (s->pb);

    ost->curidx = ogg->curidx;

    ost->next = ogg->state;

    ost->nstreams = ogg->nstreams;

    memcpy(ost->streams, ogg->streams, ogg->nstreams * sizeof(*ogg->streams));



    for (i = 0; i < ogg->nstreams; i++){

        struct ogg_stream *os = ogg->streams + i;

        os->buf = av_malloc (os->bufsize);

        memset (os->buf, 0, os->bufsize);

        memcpy (os->buf, ost->streams[i].buf, os->bufpos);

    }



    ogg->state = ost;



    return 0;

}
