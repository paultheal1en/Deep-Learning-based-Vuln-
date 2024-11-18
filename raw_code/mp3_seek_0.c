static int mp3_seek(AVFormatContext *s, int stream_index, int64_t timestamp,

                    int flags)

{

    MP3DecContext *mp3 = s->priv_data;

    AVIndexEntry *ie, ie1;

    AVStream *st = s->streams[0];

    int64_t ret  = av_index_search_timestamp(st, timestamp, flags);

    int64_t best_pos;

    int fast_seek = (s->flags & AVFMT_FLAG_FAST_SEEK) ? 1 : 0;

    int64_t filesize = mp3->header_filesize;



    if (mp3->usetoc == 2)

        return -1; // generic index code



    if (filesize <= 0) {

        int64_t size = avio_size(s->pb);

        if (size > 0 && size > s->internal->data_offset)

            filesize = size - s->internal->data_offset;

    }



    if (   (mp3->is_cbr || fast_seek)

        && (mp3->usetoc == 0 || !mp3->xing_toc)

        && st->duration > 0

        && filesize > 0) {

        ie = &ie1;

        timestamp = av_clip64(timestamp, 0, st->duration);

        ie->timestamp = timestamp;

        ie->pos       = av_rescale(timestamp, filesize, st->duration) + s->internal->data_offset;

    } else if (mp3->xing_toc) {

        if (ret < 0)

            return ret;



        ie = &st->index_entries[ret];

    } else {

        return -1;

    }



    best_pos = mp3_sync(s, ie->pos, flags);

    if (best_pos < 0)

        return best_pos;



    if (mp3->is_cbr && ie == &ie1 && mp3->frames) {

        int frame_duration = av_rescale(st->duration, 1, mp3->frames);

        ie1.timestamp = frame_duration * av_rescale(best_pos - s->internal->data_offset, mp3->frames, mp3->header_filesize);

    }



    ff_update_cur_dts(s, st, ie->timestamp);

    return 0;

}
