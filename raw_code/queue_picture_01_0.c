static int queue_picture(VideoState *is, AVFrame *src_frame, double pts, int64_t pos)

{

    VideoPicture *vp;

#if CONFIG_AVFILTER

    AVPicture pict_src;

#else

    int dst_pix_fmt = AV_PIX_FMT_YUV420P;

#endif

    /* wait until we have space to put a new picture */

    SDL_LockMutex(is->pictq_mutex);



    if (is->pictq_size >= VIDEO_PICTURE_QUEUE_SIZE && !is->refresh)

        is->skip_frames = FFMAX(1.0 - FRAME_SKIP_FACTOR, is->skip_frames * (1.0 - FRAME_SKIP_FACTOR));



    while (is->pictq_size >= VIDEO_PICTURE_QUEUE_SIZE &&

           !is->videoq.abort_request) {

        SDL_CondWait(is->pictq_cond, is->pictq_mutex);

    }

    SDL_UnlockMutex(is->pictq_mutex);



    if (is->videoq.abort_request)

        return -1;



    vp = &is->pictq[is->pictq_windex];



    vp->sar = src_frame->sample_aspect_ratio;



    /* alloc or resize hardware picture buffer */

    if (!vp->bmp || vp->reallocate ||

#if CONFIG_AVFILTER

        vp->width  != is->out_video_filter->inputs[0]->w ||

        vp->height != is->out_video_filter->inputs[0]->h) {

#else

        vp->width != is->video_st->codec->width ||

        vp->height != is->video_st->codec->height) {

#endif

        SDL_Event event;



        vp->allocated  = 0;

        vp->reallocate = 0;



        /* the allocation must be done in the main thread to avoid

           locking problems */

        event.type = FF_ALLOC_EVENT;

        event.user.data1 = is;

        SDL_PushEvent(&event);



        /* wait until the picture is allocated */

        SDL_LockMutex(is->pictq_mutex);

        while (!vp->allocated && !is->videoq.abort_request) {

            SDL_CondWait(is->pictq_cond, is->pictq_mutex);

        }

        SDL_UnlockMutex(is->pictq_mutex);



        if (is->videoq.abort_request)

            return -1;

    }



    /* if the frame is not skipped, then display it */

    if (vp->bmp) {

        AVPicture pict = { { 0 } };



        /* get a pointer on the bitmap */

        SDL_LockYUVOverlay (vp->bmp);



        pict.data[0] = vp->bmp->pixels[0];

        pict.data[1] = vp->bmp->pixels[2];

        pict.data[2] = vp->bmp->pixels[1];



        pict.linesize[0] = vp->bmp->pitches[0];

        pict.linesize[1] = vp->bmp->pitches[2];

        pict.linesize[2] = vp->bmp->pitches[1];



#if CONFIG_AVFILTER

        pict_src.data[0] = src_frame->data[0];

        pict_src.data[1] = src_frame->data[1];

        pict_src.data[2] = src_frame->data[2];



        pict_src.linesize[0] = src_frame->linesize[0];

        pict_src.linesize[1] = src_frame->linesize[1];

        pict_src.linesize[2] = src_frame->linesize[2];



        // FIXME use direct rendering

        av_picture_copy(&pict, &pict_src,

                        vp->pix_fmt, vp->width, vp->height);

#else

        av_opt_get_int(sws_opts, "sws_flags", 0, &sws_flags);

        is->img_convert_ctx = sws_getCachedContext(is->img_convert_ctx,

            vp->width, vp->height, vp->pix_fmt, vp->width, vp->height,

            dst_pix_fmt, sws_flags, NULL, NULL, NULL);

        if (is->img_convert_ctx == NULL) {

            fprintf(stderr, "Cannot initialize the conversion context\n");

            exit(1);

        }

        sws_scale(is->img_convert_ctx, src_frame->data, src_frame->linesize,

                  0, vp->height, pict.data, pict.linesize);

#endif

        /* update the bitmap content */

        SDL_UnlockYUVOverlay(vp->bmp);



        vp->pts = pts;

        vp->pos = pos;



        /* now we can update the picture count */

        if (++is->pictq_windex == VIDEO_PICTURE_QUEUE_SIZE)

            is->pictq_windex = 0;

        SDL_LockMutex(is->pictq_mutex);

        vp->target_clock = compute_target_time(vp->pts, is);



        is->pictq_size++;

        SDL_UnlockMutex(is->pictq_mutex);

    }

    return 0;

}
