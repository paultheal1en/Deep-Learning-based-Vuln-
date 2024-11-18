void *s1d13745_init(qemu_irq gpio_int)

{

    BlizzardState *s = (BlizzardState *) g_malloc0(sizeof(*s));

    DisplaySurface *surface;



    s->fb = g_malloc(0x180000);



    s->con = graphic_console_init(blizzard_update_display,

                                  blizzard_invalidate_display,

                                  blizzard_screen_dump, NULL, s);

    surface = qemu_console_surface(s->con);



    switch (surface_bits_per_pixel(surface)) {

    case 0:

        s->line_fn_tab[0] = s->line_fn_tab[1] =

                g_malloc0(sizeof(blizzard_fn_t) * 0x10);

        break;

    case 8:

        s->line_fn_tab[0] = blizzard_draw_fn_8;

        s->line_fn_tab[1] = blizzard_draw_fn_r_8;

        break;

    case 15:

        s->line_fn_tab[0] = blizzard_draw_fn_15;

        s->line_fn_tab[1] = blizzard_draw_fn_r_15;

        break;

    case 16:

        s->line_fn_tab[0] = blizzard_draw_fn_16;

        s->line_fn_tab[1] = blizzard_draw_fn_r_16;

        break;

    case 24:

        s->line_fn_tab[0] = blizzard_draw_fn_24;

        s->line_fn_tab[1] = blizzard_draw_fn_r_24;

        break;

    case 32:

        s->line_fn_tab[0] = blizzard_draw_fn_32;

        s->line_fn_tab[1] = blizzard_draw_fn_r_32;

        break;

    default:

        fprintf(stderr, "%s: Bad color depth\n", __FUNCTION__);

        exit(1);

    }



    blizzard_reset(s);



    return s;

}
