void qemu_spice_display_switch(SimpleSpiceDisplay *ssd,

                               DisplaySurface *surface)

{

    SimpleSpiceUpdate *update;

    bool need_destroy;



    if (surface && ssd->surface &&

        surface_width(surface) == pixman_image_get_width(ssd->surface) &&

        surface_height(surface) == pixman_image_get_height(ssd->surface)) {

        /* no-resize fast path: just swap backing store */

        dprint(1, "%s/%d: fast (%dx%d)\n", __func__, ssd->qxl.id,

               surface_width(surface), surface_height(surface));

        qemu_mutex_lock(&ssd->lock);

        ssd->ds = surface;

        pixman_image_unref(ssd->surface);

        ssd->surface = pixman_image_ref(ssd->ds->image);

        qemu_mutex_unlock(&ssd->lock);

        qemu_spice_display_update(ssd, 0, 0,

                                  surface_width(surface),

                                  surface_height(surface));

        return;

    }



    /* full mode switch */

    dprint(1, "%s/%d: full (%dx%d -> %dx%d)\n", __func__, ssd->qxl.id,

           ssd->surface ? pixman_image_get_width(ssd->surface)  : 0,

           ssd->surface ? pixman_image_get_height(ssd->surface) : 0,

           surface ? surface_width(surface)  : 0,

           surface ? surface_height(surface) : 0);



    memset(&ssd->dirty, 0, sizeof(ssd->dirty));

    if (ssd->surface) {

        pixman_image_unref(ssd->surface);

        ssd->surface = NULL;

        pixman_image_unref(ssd->mirror);

        ssd->mirror = NULL;

    }



    qemu_mutex_lock(&ssd->lock);

    need_destroy = (ssd->ds != NULL);

    ssd->ds = surface;

    while ((update = QTAILQ_FIRST(&ssd->updates)) != NULL) {

        QTAILQ_REMOVE(&ssd->updates, update, next);

        qemu_spice_destroy_update(ssd, update);

    }

    qemu_mutex_unlock(&ssd->lock);

    if (need_destroy) {

        qemu_spice_destroy_host_primary(ssd);

    }

    if (ssd->ds) {

        ssd->surface = pixman_image_ref(ssd->ds->image);

        ssd->mirror  = qemu_pixman_mirror_create(ssd->ds->format,

                                                 ssd->ds->image);

        qemu_spice_create_host_primary(ssd);

    }



    memset(&ssd->dirty, 0, sizeof(ssd->dirty));

    ssd->notify++;

}
