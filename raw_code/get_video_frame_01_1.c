static int get_video_frame(VideoState *is, AVFrame *frame, int64_t *pts, AVPacket *pkt, int *serial)

{

    int got_picture;



    if (packet_queue_get(&is->videoq, pkt, 1, serial) < 0)

        return -1;



    if (pkt->data == flush_pkt.data) {

        avcodec_flush_buffers(is->video_st->codec);



        SDL_LockMutex(is->pictq_mutex);

        // Make sure there are no long delay timers (ideally we should just flush the queue but that's harder)

        while (is->pictq_size && !is->videoq.abort_request) {

            SDL_CondWait(is->pictq_cond, is->pictq_mutex);

        }

        is->video_current_pos = -1;

        is->frame_last_pts = AV_NOPTS_VALUE;

        is->frame_last_duration = 0;

        is->frame_timer = (double)av_gettime() / 1000000.0;

        is->frame_last_dropped_pts = AV_NOPTS_VALUE;

        SDL_UnlockMutex(is->pictq_mutex);

        return 0;

    }



    if(avcodec_decode_video2(is->video_st->codec, frame, &got_picture, pkt) < 0)

        return 0;



    if (got_picture) {

        int ret = 1;



        if (decoder_reorder_pts == -1) {

            *pts = av_frame_get_best_effort_timestamp(frame);

        } else if (decoder_reorder_pts) {

            *pts = frame->pkt_pts;

        } else {

            *pts = frame->pkt_dts;

        }



        if (*pts == AV_NOPTS_VALUE) {

            *pts = 0;

        }



        if (framedrop>0 || (framedrop && get_master_sync_type(is) != AV_SYNC_VIDEO_MASTER)) {

            SDL_LockMutex(is->pictq_mutex);

            if (is->frame_last_pts != AV_NOPTS_VALUE && *pts) {

                double clockdiff = get_video_clock(is) - get_master_clock(is);

                double dpts = av_q2d(is->video_st->time_base) * *pts;

                double ptsdiff = dpts - is->frame_last_pts;

                if (!isnan(clockdiff) && fabs(clockdiff) < AV_NOSYNC_THRESHOLD &&

                     ptsdiff > 0 && ptsdiff < AV_NOSYNC_THRESHOLD &&

                     clockdiff + ptsdiff - is->frame_last_filter_delay < 0) {

                    is->frame_last_dropped_pos = pkt->pos;

                    is->frame_last_dropped_pts = dpts;

                    is->frame_drops_early++;


                    ret = 0;

                }

            }

            SDL_UnlockMutex(is->pictq_mutex);

        }



        return ret;

    }

    return 0;

}