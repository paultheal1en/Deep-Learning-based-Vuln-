static void sdl_grab_start(void)

{

    if (guest_cursor) {

        SDL_SetCursor(guest_sprite);

        SDL_WarpMouse(guest_x, guest_y);

    } else

        sdl_hide_cursor();



    if (SDL_WM_GrabInput(SDL_GRAB_ON) == SDL_GRAB_ON) {

        gui_grab = 1;

        sdl_update_caption();

    } else

        sdl_show_cursor();

}
