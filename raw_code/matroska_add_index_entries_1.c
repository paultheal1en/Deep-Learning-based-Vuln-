static void matroska_add_index_entries(MatroskaDemuxContext *matroska)

{

    EbmlList *index_list;

    MatroskaIndex *index;

    int index_scale = 1;

    int i, j;



    if (matroska->ctx->flags & AVFMT_FLAG_IGNIDX)

        return;



    index_list = &matroska->index;

    index      = index_list->elem;

    if (index_list->nb_elem &&

        index[0].time > 1E14 / matroska->time_scale) {

        av_log(matroska->ctx, AV_LOG_WARNING, "Working around broken index.\n");

        index_scale = matroska->time_scale;

    }

    for (i = 0; i < index_list->nb_elem; i++) {

        EbmlList *pos_list    = &index[i].pos;

        MatroskaIndexPos *pos = pos_list->elem;

        for (j = 0; j < pos_list->nb_elem; j++) {

            MatroskaTrack *track = matroska_find_track_by_num(matroska,

                                                              pos[j].track);

            if (track && track->stream)

                av_add_index_entry(track->stream,

                                   pos[j].pos + matroska->segment_start,

                                   index[i].time / index_scale, 0, 0,

                                   AVINDEX_KEYFRAME);

        }

    }

}
