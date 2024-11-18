static void sdl_mouse_mode_change(Notifier *notify, void *data)

{

    if (kbd_mouse_is_absolute()) {

        if (!absolute_enabled) {

            sdl_hide_cursor();

            if (gui_grab) {

                sdl_grab_end();

            }

            absolute_enabled = 1;

        }

    } else if (absolute_enabled) {

	sdl_show_cursor();

	absolute_enabled = 0;

    }

}
