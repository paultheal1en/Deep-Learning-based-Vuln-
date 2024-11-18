static void vga_screen_dump_common(VGAState *s, const char *filename,

                                   int w, int h)

{

    DisplayState *saved_ds, ds1, *ds = &ds1;

    DisplayChangeListener dcl;



    /* XXX: this is a little hackish */

    vga_invalidate_display(s);

    saved_ds = s->ds;



    memset(ds, 0, sizeof(DisplayState));

    memset(&dcl, 0, sizeof(DisplayChangeListener));

    dcl.dpy_update = vga_save_dpy_update;

    dcl.dpy_resize = vga_save_dpy_resize;

    dcl.dpy_refresh = vga_save_dpy_refresh;

    register_displaychangelistener(ds, &dcl);


    ds->surface = qemu_create_displaysurface(ds, w, h);



    s->ds = ds;

    s->graphic_mode = -1;

    vga_update_display(s);



    ppm_save(filename, ds->surface);



    qemu_free_displaysurface(ds);

    s->ds = saved_ds;

}