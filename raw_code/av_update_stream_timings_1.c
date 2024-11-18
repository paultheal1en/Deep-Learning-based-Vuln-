static void av_update_stream_timings(AVFormatContext *ic)

{

    int64_t start_time, start_time1, end_time, end_time1;

    int64_t duration, duration1;

    int i;

    AVStream *st;



    start_time = INT64_MAX;

    end_time = INT64_MIN;

    duration = INT64_MIN;

    for(i = 0;i < ic->nb_streams; i++) {

        st = ic->streams[i];

        if (st->start_time != AV_NOPTS_VALUE) {

            start_time1= av_rescale_q(st->start_time, st->time_base, AV_TIME_BASE_Q);

            if (start_time1 < start_time)

                start_time = start_time1;

            if (st->duration != AV_NOPTS_VALUE) {

                end_time1 = start_time1

                          + av_rescale_q(st->duration, st->time_base, AV_TIME_BASE_Q);

                if (end_time1 > end_time)

                    end_time = end_time1;

            }

        }

        if (st->duration != AV_NOPTS_VALUE) {

            duration1 = av_rescale_q(st->duration, st->time_base, AV_TIME_BASE_Q);

            if (duration1 > duration)

                duration = duration1;

        }

    }

    if (start_time != INT64_MAX) {

        ic->start_time = start_time;

        if (end_time != INT64_MIN) {

            if (end_time - start_time > duration)

                duration = end_time - start_time;

        }

    }

    if (duration != INT64_MIN) {

        ic->duration = duration;

        if (ic->file_size > 0) {

            /* compute the bitrate */

            ic->bit_rate = (double)ic->file_size * 8.0 * AV_TIME_BASE /

                (double)ic->duration;

        }

    }

}
