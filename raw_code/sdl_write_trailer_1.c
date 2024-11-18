static int sdl_write_trailer(AVFormatContext *s)

{

    SDLContext *sdl = s->priv_data;



    sdl->quit = 1;



    if (sdl->overlay)

        SDL_FreeYUVOverlay(sdl->overlay);


    if (sdl->event_thread)

        SDL_WaitThread(sdl->event_thread, NULL);


    if (sdl->mutex)

        SDL_DestroyMutex(sdl->mutex);


    if (sdl->init_cond)

        SDL_DestroyCond(sdl->init_cond);




    if (!sdl->sdl_was_already_inited)

        SDL_Quit();



    return 0;

}