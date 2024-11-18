static int hds_flush(AVFormatContext *s, OutputStream *os, int final,

                     int64_t end_ts)

{

    HDSContext *c = s->priv_data;

    int i, ret = 0;

    char target_filename[1024];

    int index = s->streams[os->first_stream]->id;



    if (!os->packets_written)

        return 0;



    avio_flush(os->ctx->pb);

    os->packets_written = 0;

    close_file(os);



    snprintf(target_filename, sizeof(target_filename),

             "%s/stream%dSeg1-Frag%d", s->filename, index, os->fragment_index);

    ret = ff_rename(os->temp_filename, target_filename);

    if (ret < 0)

        return ret;

    add_fragment(os, target_filename, os->frag_start_ts, end_ts - os->frag_start_ts);



    if (!final) {

        ret = init_file(s, os, end_ts);

        if (ret < 0)

            return ret;

    }



    if (c->window_size || (final && c->remove_at_exit)) {

        int remove = os->nb_fragments - c->window_size - c->extra_window_size;

        if (final && c->remove_at_exit)

            remove = os->nb_fragments;

        if (remove > 0) {

            for (i = 0; i < remove; i++) {

                unlink(os->fragments[i]->file);

                av_free(os->fragments[i]);

            }

            os->nb_fragments -= remove;

            memmove(os->fragments, os->fragments + remove,

                    os->nb_fragments * sizeof(*os->fragments));

        }

    }



    if (ret >= 0)

        ret = write_abst(s, os, final);

    return ret;

}
