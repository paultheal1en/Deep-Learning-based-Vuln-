static void handle_keyup(DisplayState *ds, SDL_Event *ev)

{

    int mod_state;



    if (!alt_grab) {

        mod_state = (ev->key.keysym.mod & gui_grab_code);

    } else {

        mod_state = (ev->key.keysym.mod & (gui_grab_code | KMOD_LSHIFT));

    }

    if (!mod_state && gui_key_modifier_pressed) {

        gui_key_modifier_pressed = 0;

        if (gui_keysym == 0) {

            /* exit/enter grab if pressing Ctrl-Alt */

            if (!gui_grab) {

                /* If the application is not active, do not try to enter grab

                 * state. It prevents 'SDL_WM_GrabInput(SDL_GRAB_ON)' from

                 * blocking all the application (SDL bug). */

                if (is_graphic_console() &&

                    SDL_GetAppState() & SDL_APPACTIVE) {

                    sdl_grab_start();

                }

            } else if (!gui_fullscreen) {

                sdl_grab_end();

            }

            /* SDL does not send back all the modifiers key, so we must

             * correct it. */

            reset_keys();

            return;

        }

        gui_keysym = 0;

    }

    if (is_graphic_console() && !gui_keysym) {

        sdl_process_key(&ev->key);

    }

}
