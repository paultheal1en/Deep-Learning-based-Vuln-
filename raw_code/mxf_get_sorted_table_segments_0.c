static int mxf_get_sorted_table_segments(MXFContext *mxf, int *nb_sorted_segments, MXFIndexTableSegment ***sorted_segments)

{

    int i, j, nb_segments = 0;

    MXFIndexTableSegment **unsorted_segments;

    int last_body_sid = -1, last_index_sid = -1, last_index_start = -1;



    /* count number of segments, allocate arrays and copy unsorted segments */

    for (i = 0; i < mxf->metadata_sets_count; i++)

        if (mxf->metadata_sets[i]->type == IndexTableSegment)

            nb_segments++;



    if (!nb_segments)

        return AVERROR_INVALIDDATA;



    *sorted_segments  = av_mallocz(nb_segments * sizeof(**sorted_segments));

    unsorted_segments = av_mallocz(nb_segments * sizeof(*unsorted_segments));

    if (!sorted_segments || !unsorted_segments) {

        av_freep(sorted_segments);

        av_free(unsorted_segments);

        return AVERROR(ENOMEM);

    }



    for (i = j = 0; i < mxf->metadata_sets_count; i++)

        if (mxf->metadata_sets[i]->type == IndexTableSegment)

            unsorted_segments[j++] = (MXFIndexTableSegment*)mxf->metadata_sets[i];



    *nb_sorted_segments = 0;



    /* sort segments by {BodySID, IndexSID, IndexStartPosition}, remove duplicates while we're at it */

    for (i = 0; i < nb_segments; i++) {

        int best = -1, best_body_sid = -1, best_index_sid = -1, best_index_start = -1;



        for (j = 0; j < nb_segments; j++) {

            MXFIndexTableSegment *s = unsorted_segments[j];



            /* Require larger BosySID, IndexSID or IndexStartPosition then the previous entry. This removes duplicates.

             * We want the smallest values for the keys than what we currently have, unless this is the first such entry this time around.

             */

            if ((i == 0     || s->body_sid > last_body_sid || s->index_sid > last_index_sid || s->index_start_position > last_index_start) &&

                (best == -1 || s->body_sid < best_body_sid || s->index_sid < best_index_sid || s->index_start_position < best_index_start)) {

                best             = j;

                best_body_sid    = s->body_sid;

                best_index_sid   = s->index_sid;

                best_index_start = s->index_start_position;

            }

        }



        /* no suitable entry found -> we're done */

        if (best == -1)

            break;



        (*sorted_segments)[(*nb_sorted_segments)++] = unsorted_segments[best];

        last_body_sid    = best_body_sid;

        last_index_sid   = best_index_sid;

        last_index_start = best_index_start;

    }



    av_free(unsorted_segments);



    return 0;

}
