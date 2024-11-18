static void handle_mousemotion(DisplayState *ds, SDL_Event *ev)

{

    int max_x, max_y;



    if (is_graphic_console() &&

        (kbd_mouse_is_absolute() || absolute_enabled)) {

        max_x = real_screen->w - 1;

        max_y = real_screen->h - 1;

        if (gui_grab && (ev->motion.x == 0 || ev->motion.y == 0 ||

            ev->motion.x == max_x || ev->motion.y == max_y)) {

            sdl_grab_end();

        }

        if (!gui_grab && SDL_GetAppState() & SDL_APPINPUTFOCUS &&

            (ev->motion.x > 0 && ev->motion.x < max_x &&

            ev->motion.y > 0 && ev->motion.y < max_y)) {

            sdl_grab_start();

        }

    }

    if (gui_grab || kbd_mouse_is_absolute() || absolute_enabled) {

        sdl_send_mouse_event(ev->motion.xrel, ev->motion.yrel, 0,

                             ev->motion.x, ev->motion.y, ev->motion.state);

    }

}
