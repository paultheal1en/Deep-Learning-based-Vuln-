static void sdl_grab_start(void)

{

    /*

     * If the application is not active, do not try to enter grab state. This

     * prevents 'SDL_WM_GrabInput(SDL_GRAB_ON)' from blocking all the

     * application (SDL bug).

     */

    if (!(SDL_GetAppState() & SDL_APPINPUTFOCUS)) {

        return;

    }

    if (guest_cursor) {

        SDL_SetCursor(guest_sprite);

        if (!kbd_mouse_is_absolute() && !absolute_enabled)

            SDL_WarpMouse(guest_x, guest_y);

    } else

        sdl_hide_cursor();



    if (SDL_WM_GrabInput(SDL_GRAB_ON) == SDL_GRAB_ON) {

        gui_grab = 1;

        sdl_update_caption();

    } else

        sdl_show_cursor();

}
