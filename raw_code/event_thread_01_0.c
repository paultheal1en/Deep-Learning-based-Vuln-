static int event_thread(void *arg)

{

    AVFormatContext *s = arg;

    SDLContext *sdl = s->priv_data;

    int flags = SDL_BASE_FLAGS | (sdl->window_fullscreen ? SDL_FULLSCREEN : 0);

    AVStream *st = s->streams[0];

    AVCodecContext *encctx = st->codec;



    /* initialization */

    if (SDL_Init(SDL_INIT_VIDEO) != 0) {

        av_log(s, AV_LOG_ERROR, "Unable to initialize SDL: %s\n", SDL_GetError());

        sdl->init_ret = AVERROR(EINVAL);

        goto init_end;

    }



    SDL_WM_SetCaption(sdl->window_title, sdl->icon_title);

    sdl->surface = SDL_SetVideoMode(sdl->window_width, sdl->window_height,

                                    24, flags);

    if (!sdl->surface) {

        av_log(sdl, AV_LOG_ERROR, "Unable to set video mode: %s\n", SDL_GetError());

        sdl->init_ret = AVERROR(EINVAL);

        goto init_end;

    }



    sdl->overlay = SDL_CreateYUVOverlay(encctx->width, encctx->height,

                                        sdl->overlay_fmt, sdl->surface);

    if (!sdl->overlay || sdl->overlay->pitches[0] < encctx->width) {

        av_log(s, AV_LOG_ERROR,

               "SDL does not support an overlay with size of %dx%d pixels\n",

               encctx->width, encctx->height);

        sdl->init_ret = AVERROR(EINVAL);

        goto init_end;

    }



    sdl->init_ret = 0;

    av_log(s, AV_LOG_VERBOSE, "w:%d h:%d fmt:%s -> w:%d h:%d\n",

           encctx->width, encctx->height, av_get_pix_fmt_name(encctx->pix_fmt),

           sdl->overlay_rect.w, sdl->overlay_rect.h);



init_end:

    SDL_LockMutex(sdl->mutex);

    sdl->inited = 1;

    SDL_UnlockMutex(sdl->mutex);

    SDL_CondSignal(sdl->init_cond);



    if (sdl->init_ret < 0)

        return sdl->init_ret;



    /* event loop */

    while (!sdl->quit) {

        int ret;

        SDL_Event event;

        SDL_PumpEvents();

        ret = SDL_PeepEvents(&event, 1, SDL_GETEVENT, SDL_ALLEVENTS);

        if (ret < 0)

            av_log(s, AV_LOG_ERROR, "Error when getting SDL event: %s\n", SDL_GetError());

        if (ret <= 0)

            continue;



        switch (event.type) {

        case SDL_KEYDOWN:

            switch (event.key.keysym.sym) {

            case SDLK_ESCAPE:

            case SDLK_q:

                sdl->quit = 1;

                break;

            }

            break;

        case SDL_QUIT:

            sdl->quit = 1;

            break;



        case SDL_VIDEORESIZE:

            sdl->window_width  = event.resize.w;

            sdl->window_height = event.resize.h;



            SDL_LockMutex(sdl->mutex);

            sdl->surface = SDL_SetVideoMode(sdl->window_width, sdl->window_height, 24, SDL_BASE_FLAGS);

            if (!sdl->surface) {

                av_log(s, AV_LOG_ERROR, "Failed to set SDL video mode: %s\n", SDL_GetError());

                sdl->quit = 1;

            } else {

                compute_overlay_rect(s);

            }

            SDL_UnlockMutex(sdl->mutex);

            break;



        default:

            break;

        }

    }



    return 0;

}
