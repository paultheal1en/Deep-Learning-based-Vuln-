static void stream_pause(VideoState *is)

{

    is->paused = !is->paused;

    if (!is->paused) {

        if(is->read_pause_return != AVERROR(ENOSYS)){

            is->video_current_pts = get_video_clock(is);

        }



        is->frame_timer += (av_gettime() - is->video_current_pts_time) / 1000000.0;

        is->video_current_pts_time= av_gettime();

    }

}
