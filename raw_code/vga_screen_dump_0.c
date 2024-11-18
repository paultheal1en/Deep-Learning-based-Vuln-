static void vga_screen_dump(void *opaque, const char *filename, bool cswitch,

                            Error **errp)

{

    VGACommonState *s = opaque;

    DisplaySurface *surface = qemu_console_surface(s->con);



    if (cswitch) {

        vga_invalidate_display(s);

    }

    graphic_hw_update(s->con);

    ppm_save(filename, surface, errp);

}
