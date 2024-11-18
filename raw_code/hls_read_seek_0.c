static int hls_read_seek(AVFormatContext *s, int stream_index,

                               int64_t timestamp, int flags)

{

    HLSContext *c = s->priv_data;

    int i;

    int64_t seek_timestamp;

    int valid_for = -1;



    if ((flags & AVSEEK_FLAG_BYTE) || !c->variants[0]->playlists[0]->finished)

        return AVERROR(ENOSYS);



    seek_timestamp = stream_index < 0 ? timestamp :

                     av_rescale_rnd(timestamp, AV_TIME_BASE,

                                    s->streams[stream_index]->time_base.den,

                                    flags & AVSEEK_FLAG_BACKWARD ?

                                    AV_ROUND_DOWN : AV_ROUND_UP);



    if (s->duration < seek_timestamp)

        return AVERROR(EIO);



    for (i = 0; i < c->n_playlists; i++) {

        /* check first that the timestamp is valid for some playlist */

        struct playlist *pls = c->playlists[i];

        int seq_no;

        if (find_timestamp_in_playlist(c, pls, seek_timestamp, &seq_no)) {

            /* set segment now so we do not need to search again below */

            pls->cur_seq_no = seq_no;

            valid_for = i;

            break;

        }

    }



    if (valid_for < 0)

        return AVERROR(EIO);



    for (i = 0; i < c->n_playlists; i++) {

        /* Reset reading */

        struct playlist *pls = c->playlists[i];

        if (pls->input) {

            ffurl_close(pls->input);

            pls->input = NULL;

        }

        av_free_packet(&pls->pkt);

        reset_packet(&pls->pkt);

        pls->pb.eof_reached = 0;

        /* Clear any buffered data */

        pls->pb.buf_end = pls->pb.buf_ptr = pls->pb.buffer;

        /* Reset the pos, to let the mpegts demuxer know we've seeked. */

        pls->pb.pos = 0;



        pls->seek_timestamp = seek_timestamp;

        pls->seek_flags = flags;



        /* set closest segment seq_no for playlists not handled above */

        if (valid_for != i)

            find_timestamp_in_playlist(c, pls, seek_timestamp, &pls->cur_seq_no);

    }



    c->cur_timestamp = seek_timestamp;



    return 0;

}
