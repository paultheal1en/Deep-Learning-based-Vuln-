static int refresh_thread(void *opaque)

{

    VideoState *is= opaque;

    while (!is->abort_request) {

        SDL_Event event;

        event.type = FF_REFRESH_EVENT;

        event.user.data1 = opaque;

        if (!is->refresh && (!is->paused || is->force_refresh)) {

            is->refresh = 1;

            SDL_PushEvent(&event);

        }

        //FIXME ideally we should wait the correct time but SDLs event passing is so slow it would be silly

        av_usleep(is->audio_st && is->show_mode != SHOW_MODE_VIDEO ? rdftspeed*1000 : 5000);

    }

    return 0;

}
