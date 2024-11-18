static void update_video_pts(VideoState *is, double pts, int64_t pos, int serial) {

    double time = av_gettime() / 1000000.0;

    /* update current video pts */

    is->video_current_pts = pts;

    is->video_current_pts_drift = is->video_current_pts - time;

    is->video_current_pos = pos;

    is->frame_last_pts = pts;

    check_external_clock_sync(is, is->video_current_pts);

}
