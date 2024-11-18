void qemu_spice_display_init(DisplayState *ds)

{

    assert(sdpy.ds == NULL);

    sdpy.ds = ds;

    sdpy.bufsize = (16 * 1024 * 1024);

    sdpy.buf = qemu_malloc(sdpy.bufsize);

    pthread_mutex_init(&sdpy.lock, NULL);

    register_displaychangelistener(ds, &display_listener);



    sdpy.qxl.base.sif = &dpy_interface.base;

    qemu_spice_add_interface(&sdpy.qxl.base);

    assert(sdpy.worker);



    qemu_add_vm_change_state_handler(qemu_spice_vm_change_state_handler, &sdpy);

    qemu_spice_create_host_memslot(&sdpy);

    qemu_spice_create_host_primary(&sdpy);

}
