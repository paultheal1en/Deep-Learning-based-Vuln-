void qemu_spice_display_init(DisplayState *ds)

{

    assert(sdpy.ds == NULL);

    qemu_spice_display_init_common(&sdpy, ds);

    register_displaychangelistener(ds, &display_listener);



    sdpy.qxl.base.sif = &dpy_interface.base;

    qemu_spice_add_interface(&sdpy.qxl.base);

    assert(sdpy.worker);



    qemu_spice_create_host_memslot(&sdpy);

    qemu_spice_create_host_primary(&sdpy);

}
