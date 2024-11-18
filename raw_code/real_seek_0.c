static int real_seek(AVFormatContext *avf, int stream,

                     int64_t min_ts, int64_t ts, int64_t max_ts, int flags)

{

    ConcatContext *cat = avf->priv_data;

    int ret, left, right;



    if (stream >= 0) {

        if (stream >= avf->nb_streams)

            return AVERROR(EINVAL);

        rescale_interval(avf->streams[stream]->time_base, AV_TIME_BASE_Q,

                         &min_ts, &ts, &max_ts);

    }



    left  = 0;

    right = cat->nb_files;

    while (right - left > 1) {

        int mid = (left + right) / 2;

        if (ts < cat->files[mid].start_time)

            right = mid;

        else

            left  = mid;

    }



    if ((ret = open_file(avf, left)) < 0)

        return ret;



    ret = try_seek(avf, stream, min_ts, ts, max_ts, flags);

    if (ret < 0 && !(flags & AVSEEK_FLAG_BACKWARD) &&

        left < cat->nb_files - 1 &&

        cat->files[left + 1].start_time < max_ts) {

        if ((ret = open_file(avf, left + 1)) < 0)

            return ret;

        ret = try_seek(avf, stream, min_ts, ts, max_ts, flags);

    }

    return ret;

}
