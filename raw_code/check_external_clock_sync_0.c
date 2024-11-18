static void check_external_clock_sync(VideoState *is, double pts) {

    if (fabs(get_external_clock(is) - pts) > AV_NOSYNC_THRESHOLD) {

        update_external_clock_pts(is, pts);

    }

}
