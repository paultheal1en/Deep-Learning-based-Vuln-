static int get_video_frame(VideoState *is, AVFrame *frame, int64_t *pts, AVPacket *pkt)

{

    int got_picture, i;



    if (packet_queue_get(&is->videoq, pkt, 1) < 0)

        return -1;



    if (pkt->data == flush_pkt.data) {

        avcodec_flush_buffers(is->video_st->codec);



        SDL_LockMutex(is->pictq_mutex);

        // Make sure there are no long delay timers (ideally we should just flush the que but thats harder)

        for (i = 0; i < VIDEO_PICTURE_QUEUE_SIZE; i++) {

            is->pictq[i].skip = 1;

        }

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



    avcodec_decode_video2(is->video_st->codec, frame, &got_picture, pkt);



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



        if (((is->av_sync_type == AV_SYNC_AUDIO_MASTER && is->audio_st) || is->av_sync_type == AV_SYNC_EXTERNAL_CLOCK) &&

             (framedrop>0 || (framedrop && is->audio_st))) {

            SDL_LockMutex(is->pictq_mutex);

            if (is->frame_last_pts != AV_NOPTS_VALUE && *pts) {

                double clockdiff = get_video_clock(is) - get_master_clock(is);

                double dpts = av_q2d(is->video_st->time_base) * *pts;

                double ptsdiff = dpts - is->frame_last_pts;

                if (fabs(clockdiff) < AV_NOSYNC_THRESHOLD &&

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
