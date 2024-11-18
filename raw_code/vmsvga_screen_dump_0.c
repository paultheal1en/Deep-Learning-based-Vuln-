static void vmsvga_screen_dump(void *opaque, const char *filename, bool cswitch,

                               Error **errp)

{

    struct vmsvga_state_s *s = opaque;

    DisplaySurface *surface = qemu_console_surface(s->vga.con);



    if (!s->enable) {

        s->vga.screen_dump(&s->vga, filename, cswitch, errp);

        return;

    }



    if (surface_bits_per_pixel(surface) == 32) {

        DisplaySurface *ds = qemu_create_displaysurface_from(

                                 surface_width(surface),

                                 surface_height(surface),

                                 32,

                                 surface_stride(surface),

                                 s->vga.vram_ptr, false);

        ppm_save(filename, ds, errp);

        g_free(ds);

    }

}
