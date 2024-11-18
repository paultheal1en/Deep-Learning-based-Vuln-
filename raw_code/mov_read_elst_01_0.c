static int mov_read_elst(MOVContext *c, ByteIOContext *pb, MOVAtom atom)

{

    MOVStreamContext *sc = c->fc->streams[c->fc->nb_streams-1]->priv_data;

    int i, edit_count;



    get_byte(pb); /* version */

    get_be24(pb); /* flags */

    edit_count= sc->edit_count = get_be32(pb);     /* entries */



    for(i=0; i<edit_count; i++){

        int time;

        get_be32(pb); /* Track duration */

        time = get_be32(pb); /* Media time */

        get_be32(pb); /* Media rate */

        if (time != 0)

            av_log(c->fc, AV_LOG_WARNING, "edit list not starting at 0, "

                   "a/v desync might occur, patch welcome\n");

    }

    dprintf(c->fc, "track[%i].edit_count = %i\n", c->fc->nb_streams-1, sc->edit_count);

    return 0;

}
