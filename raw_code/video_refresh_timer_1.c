static void video_refresh_timer(void *opaque)

{

    VideoState *is = opaque;

    VideoPicture *vp;



    SubPicture *sp, *sp2;



    if (is->video_st) {

        if (is->pictq_size == 0) {

            /* if no picture, need to wait */

            schedule_refresh(is, 1);

        } else {

            /* dequeue the picture */

            vp = &is->pictq[is->pictq_rindex];



            /* update current video pts */

            is->video_current_pts = vp->pts;

            is->video_current_pts_time = av_gettime();



            /* launch timer for next picture */

            schedule_refresh(is, (int)(compute_frame_delay(vp->pts, is) * 1000 + 0.5));



            if(is->subtitle_st) {

                if (is->subtitle_stream_changed) {

                    SDL_LockMutex(is->subpq_mutex);



                    while (is->subpq_size) {

                        free_subpicture(&is->subpq[is->subpq_rindex]);



                        /* update queue size and signal for next picture */

                        if (++is->subpq_rindex == SUBPICTURE_QUEUE_SIZE)

                            is->subpq_rindex = 0;



                        is->subpq_size--;

                    }

                    is->subtitle_stream_changed = 0;



                    SDL_CondSignal(is->subpq_cond);

                    SDL_UnlockMutex(is->subpq_mutex);

                } else {

                    if (is->subpq_size > 0) {

                        sp = &is->subpq[is->subpq_rindex];



                        if (is->subpq_size > 1)

                            sp2 = &is->subpq[(is->subpq_rindex + 1) % SUBPICTURE_QUEUE_SIZE];

                        else

                            sp2 = NULL;



                        if ((is->video_current_pts > (sp->pts + ((float) sp->sub.end_display_time / 1000)))

                                || (sp2 && is->video_current_pts > (sp2->pts + ((float) sp2->sub.start_display_time / 1000))))

                        {

                            free_subpicture(sp);



                            /* update queue size and signal for next picture */

                            if (++is->subpq_rindex == SUBPICTURE_QUEUE_SIZE)

                                is->subpq_rindex = 0;



                            SDL_LockMutex(is->subpq_mutex);

                            is->subpq_size--;

                            SDL_CondSignal(is->subpq_cond);

                            SDL_UnlockMutex(is->subpq_mutex);

                        }

                    }

                }

            }



            /* display picture */

            video_display(is);



            /* update queue size and signal for next picture */

            if (++is->pictq_rindex == VIDEO_PICTURE_QUEUE_SIZE)

                is->pictq_rindex = 0;



            SDL_LockMutex(is->pictq_mutex);

            is->pictq_size--;

            SDL_CondSignal(is->pictq_cond);

            SDL_UnlockMutex(is->pictq_mutex);

        }

    } else if (is->audio_st) {

        /* draw the next audio frame */



        schedule_refresh(is, 40);



        /* if only audio stream, then display the audio bars (better

           than nothing, just to test the implementation */



        /* display picture */

        video_display(is);

    } else {

        schedule_refresh(is, 100);

    }

    if (show_status) {

        static int64_t last_time;

        int64_t cur_time;

        int aqsize, vqsize, sqsize;

        double av_diff;



        cur_time = av_gettime();

        if (!last_time || (cur_time - last_time) >= 30000) {

            aqsize = 0;

            vqsize = 0;

            sqsize = 0;

            if (is->audio_st)

                aqsize = is->audioq.size;

            if (is->video_st)

                vqsize = is->videoq.size;

            if (is->subtitle_st)

                sqsize = is->subtitleq.size;

            av_diff = 0;

            if (is->audio_st && is->video_st)

                av_diff = get_audio_clock(is) - get_video_clock(is);

            printf("%7.2f A-V:%7.3f aq=%5dKB vq=%5dKB sq=%5dB f=%Ld/%Ld   \r",

                   get_master_clock(is), av_diff, aqsize / 1024, vqsize / 1024, sqsize, is->faulty_dts, is->faulty_pts);

            fflush(stdout);

            last_time = cur_time;

        }

    }

}
