static void blizzard_screen_dump(void *opaque, const char *filename,

                                 bool cswitch, Error **errp)

{

    BlizzardState *s = (BlizzardState *) opaque;

    DisplaySurface *surface = qemu_console_surface(s->con);



    blizzard_update_display(opaque);

    if (s && surface_data(surface)) {

        ppm_save(filename, surface, errp);

    }

}
