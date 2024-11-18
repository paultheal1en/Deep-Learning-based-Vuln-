static int ogg_restore(AVFormatContext *s)

{

    struct ogg *ogg = s->priv_data;

    AVIOContext *bc = s->pb;

    struct ogg_state *ost = ogg->state;

    int i, err;



    if (!ost)

        return 0;



    ogg->state = ost->next;



        for (i = 0; i < ogg->nstreams; i++)

            av_freep(&ogg->streams[i].buf);



        avio_seek(bc, ost->pos, SEEK_SET);

        ogg->page_pos = -1;

        ogg->curidx   = ost->curidx;

        ogg->nstreams = ost->nstreams;

        if ((err = av_reallocp_array(&ogg->streams, ogg->nstreams,

                                     sizeof(*ogg->streams))) < 0) {

            ogg->nstreams = 0;

            return err;

        } else

            memcpy(ogg->streams, ost->streams,

                   ost->nstreams * sizeof(*ogg->streams));



    av_free(ost);



    return 0;

}
