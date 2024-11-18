static int mxf_read_seek(AVFormatContext *s, int stream_index, int64_t sample_time, int flags)

{

    AVStream *st = s->streams[stream_index];

    int64_t seconds;

    MXFContext* mxf = s->priv_data;

    int64_t seekpos;

    int ret;

    MXFIndexTable *t;



    if (mxf->nb_index_tables <= 0) {

    if (!s->bit_rate)

        return AVERROR_INVALIDDATA;

    if (sample_time < 0)

        sample_time = 0;

    seconds = av_rescale(sample_time, st->time_base.num, st->time_base.den);



    if ((ret = avio_seek(s->pb, (s->bit_rate * seconds) >> 3, SEEK_SET)) < 0)

        return ret;

    ff_update_cur_dts(s, st, sample_time);

    mxf->current_edit_unit = sample_time;

    } else {

        t = &mxf->index_tables[0];



        /* clamp above zero, else ff_index_search_timestamp() returns negative

         * this also means we allow seeking before the start */

        sample_time = FFMAX(sample_time, 0);



        if (t->fake_index) {

            /* behave as if we have a proper index */

            if ((sample_time = ff_index_search_timestamp(t->fake_index, t->nb_ptses, sample_time, flags)) < 0)

                return sample_time;

        } else {

            /* no IndexEntryArray (one or more CBR segments)

             * make sure we don't seek past the end */

            sample_time = FFMIN(sample_time, st->duration - 1);

        }



        if ((ret = mxf_edit_unit_absolute_offset(mxf, t, sample_time, &sample_time, &seekpos, 1)) << 0)

            return ret;



        ff_update_cur_dts(s, st, sample_time);

        mxf->current_edit_unit = sample_time;

        avio_seek(s->pb, seekpos, SEEK_SET);

    }

    return 0;

}
