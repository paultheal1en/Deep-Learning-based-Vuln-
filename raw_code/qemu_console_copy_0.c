void qemu_console_copy(QEMUConsole *console, int src_x, int src_y,

                int dst_x, int dst_y, int w, int h) {

    if (active_console == console) {

        if (console->ds->dpy_copy)

            console->ds->dpy_copy(console->ds,

                            src_x, src_y, dst_x, dst_y, w, h);

        else {

            /* TODO */

            console->ds->dpy_update(console->ds, dst_x, dst_y, w, h);

        }

    }

}
