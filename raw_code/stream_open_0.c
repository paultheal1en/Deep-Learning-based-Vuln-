static VideoState *stream_open(const char *filename, AVInputFormat *iformat)

{

    VideoState *is;



    is = av_mallocz(sizeof(VideoState));

    if (!is)

        return NULL;

    av_strlcpy(is->filename, filename, sizeof(is->filename));

    is->iformat = iformat;

    is->ytop    = 0;

    is->xleft   = 0;



    /* start video display */

    is->pictq_mutex = SDL_CreateMutex();

    is->pictq_cond  = SDL_CreateCond();



    is->subpq_mutex = SDL_CreateMutex();

    is->subpq_cond  = SDL_CreateCond();



    packet_queue_init(&is->videoq);

    packet_queue_init(&is->audioq);

    packet_queue_init(&is->subtitleq);



    is->continue_read_thread = SDL_CreateCond();



    //FIXME: use a cleaner way to signal obsolete external clock...

    update_external_clock_pts(is, (double)AV_NOPTS_VALUE);

    update_external_clock_speed(is, 1.0);

    is->audio_current_pts_drift = -av_gettime() / 1000000.0;

    is->video_current_pts_drift = is->audio_current_pts_drift;

    is->audio_clock_serial = -1;

    is->video_clock_serial = -1;

    is->av_sync_type = av_sync_type;

    is->read_tid     = SDL_CreateThread(read_thread, is);

    if (!is->read_tid) {

        av_free(is);

        return NULL;

    }

    return is;

}
