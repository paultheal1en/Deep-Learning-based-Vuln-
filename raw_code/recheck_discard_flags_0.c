static int recheck_discard_flags(AVFormatContext *s, int first)

{

    HLSContext *c = s->priv_data;

    int i, changed = 0;



    /* Check if any new streams are needed */

    for (i = 0; i < c->n_playlists; i++)

        c->playlists[i]->cur_needed = 0;



    for (i = 0; i < s->nb_streams; i++) {

        AVStream *st = s->streams[i];

        struct playlist *pls = c->playlists[s->streams[i]->id];

        if (st->discard < AVDISCARD_ALL)

            pls->cur_needed = 1;

    }

    for (i = 0; i < c->n_playlists; i++) {

        struct playlist *pls = c->playlists[i];

        if (pls->cur_needed && !pls->needed) {

            pls->needed = 1;

            changed = 1;

            pls->cur_seq_no = select_cur_seq_no(c, pls);

            pls->pb.eof_reached = 0;

            if (c->cur_timestamp != AV_NOPTS_VALUE) {

                /* catch up */

                pls->seek_timestamp = c->cur_timestamp;

                pls->seek_flags = AVSEEK_FLAG_ANY;

                pls->seek_stream_index = -1;

            }

            av_log(s, AV_LOG_INFO, "Now receiving playlist %d, segment %d\n", i, pls->cur_seq_no);

        } else if (first && !pls->cur_needed && pls->needed) {

            if (pls->input)

                ff_format_io_close(pls->parent, &pls->input);

            pls->needed = 0;

            changed = 1;

            av_log(s, AV_LOG_INFO, "No longer receiving playlist %d\n", i);

        }

    }

    return changed;

}
