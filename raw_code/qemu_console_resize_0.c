void qemu_console_resize(QEMUConsole *console, int width, int height)

{

    if (console->g_width != width || console->g_height != height) {

        console->g_width = width;

        console->g_height = height;

        if (active_console == console) {

            dpy_resize(console->ds, width, height);

        }

    }

}
