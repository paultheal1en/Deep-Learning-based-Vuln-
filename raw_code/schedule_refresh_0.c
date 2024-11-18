static void schedule_refresh(VideoState *is, int delay)

{

    if(!delay) delay=1; //SDL seems to be buggy when the delay is 0

    SDL_AddTimer(delay, sdl_refresh_timer_cb, is);

}
