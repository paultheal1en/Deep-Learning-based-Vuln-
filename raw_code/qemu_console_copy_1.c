void qemu_console_copy(QemuConsole *con, int src_x, int src_y,

                       int dst_x, int dst_y, int w, int h)

{

    assert(con->console_type == GRAPHIC_CONSOLE);

    dpy_gfx_copy(con, src_x, src_y, dst_x, dst_y, w, h);

}
