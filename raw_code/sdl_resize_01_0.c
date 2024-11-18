static void sdl_resize(DisplayState *ds)

{

    int flags;



    //    printf("resizing to %d %d\n", w, h);



    flags = SDL_HWSURFACE|SDL_ASYNCBLIT|SDL_HWACCEL;

    if (gui_fullscreen)

        flags |= SDL_FULLSCREEN;

    if (gui_noframe)

        flags |= SDL_NOFRAME;



 again:

    real_screen = SDL_SetVideoMode(ds_get_width(ds), ds_get_height(ds), 0, flags);

    if (!real_screen) {

        fprintf(stderr, "Could not open SDL display\n");

        exit(1);

    }

    if (!real_screen->pixels && (flags & SDL_HWSURFACE) && (flags & SDL_FULLSCREEN)) {

        flags &= ~SDL_HWSURFACE;

        goto again;

    }



    if (!real_screen->pixels) {

        fprintf(stderr, "Could not open SDL display\n");

        exit(1);

    }



    sdl_setdata(ds);

}
