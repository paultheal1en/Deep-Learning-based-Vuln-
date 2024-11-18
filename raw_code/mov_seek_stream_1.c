static int mov_seek_stream(AVFormatContext *s, AVStream *st, int64_t timestamp, int flags)

{

    MOVStreamContext *sc = st->priv_data;

    int sample, time_sample;

    int i;



    sample = av_index_search_timestamp(st, timestamp, flags);

    av_log(s, AV_LOG_TRACE, "stream %d, timestamp %"PRId64", sample %d\n", st->index, timestamp, sample);

    if (sample < 0 && st->nb_index_entries && timestamp < st->index_entries[0].timestamp)

        sample = 0;

    if (sample < 0) /* not sure what to do */

        return AVERROR_INVALIDDATA;

    sc->current_sample = sample;

    av_log(s, AV_LOG_TRACE, "stream %d, found sample %d\n", st->index, sc->current_sample);

    /* adjust ctts index */

    if (sc->ctts_data) {

        time_sample = 0;

        for (i = 0; i < sc->ctts_count; i++) {

            int next = time_sample + sc->ctts_data[i].count;

            if (next > sc->current_sample) {

                sc->ctts_index = i;

                sc->ctts_sample = sc->current_sample - time_sample;

                break;

            }

            time_sample = next;

        }

    }



    /* adjust stsd index */

    time_sample = 0;

    for (i = 0; i < sc->stsc_count; i++) {

        int next = time_sample + mov_get_stsc_samples(sc, i);

        if (next > sc->current_sample) {

            sc->stsc_index = i;

            sc->stsc_sample = sc->current_sample - time_sample;

            break;

        }

        time_sample = next;

    }



    return sample;

}
