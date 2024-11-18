void qemu_spice_display_resize(SimpleSpiceDisplay *ssd)

{

    dprint(1, "%s:\n", __FUNCTION__);



    pthread_mutex_lock(&ssd->lock);

    memset(&ssd->dirty, 0, sizeof(ssd->dirty));

    qemu_pf_conv_put(ssd->conv);

    ssd->conv = NULL;

    pthread_mutex_unlock(&ssd->lock);



    qemu_spice_destroy_host_primary(ssd);

    qemu_spice_create_host_primary(ssd);



    pthread_mutex_lock(&ssd->lock);

    memset(&ssd->dirty, 0, sizeof(ssd->dirty));

    ssd->notify++;

    pthread_mutex_unlock(&ssd->lock);

}
