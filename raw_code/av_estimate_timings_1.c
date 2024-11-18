static void av_estimate_timings(AVFormatContext *ic)

{

    int64_t file_size;



    /* get the file size, if possible */

    if (ic->iformat->flags & AVFMT_NOFILE) {

        file_size = 0;

    } else {

        file_size = url_fsize(&ic->pb);

        if (file_size < 0)

            file_size = 0;

    }

    ic->file_size = file_size;



    if ((!strcmp(ic->iformat->name, "mpeg") ||

         !strcmp(ic->iformat->name, "mpegts")) &&

        file_size && !ic->pb.is_streamed) {

        /* get accurate estimate from the PTSes */

        av_estimate_timings_from_pts(ic);

    } else if (av_has_timings(ic)) {

        /* at least one components has timings - we use them for all

           the components */

        fill_all_stream_timings(ic);

    } else {

        /* less precise: use bit rate info */

        av_estimate_timings_from_bit_rate(ic);

    }

    av_update_stream_timings(ic);



#if 0

    {

        int i;

        AVStream *st;

        for(i = 0;i < ic->nb_streams; i++) {

            st = ic->streams[i];

        printf("%d: start_time: %0.3f duration: %0.3f\n",

               i, (double)st->start_time / AV_TIME_BASE,

               (double)st->duration / AV_TIME_BASE);

        }

        printf("stream: start_time: %0.3f duration: %0.3f bitrate=%d kb/s\n",

               (double)ic->start_time / AV_TIME_BASE,

               (double)ic->duration / AV_TIME_BASE,

               ic->bit_rate / 1000);

    }

#endif

}
