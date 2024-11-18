void qemu_spice_display_update(SimpleSpiceDisplay *ssd,

                               int x, int y, int w, int h)

{

    QXLRect update_area;



    dprint(2, "%s: x %d y %d w %d h %d\n", __FUNCTION__, x, y, w, h);

    update_area.left = x,

    update_area.right = x + w;

    update_area.top = y;

    update_area.bottom = y + h;



    pthread_mutex_lock(&ssd->lock);

    if (qemu_spice_rect_is_empty(&ssd->dirty)) {

        ssd->notify++;

    }

    qemu_spice_rect_union(&ssd->dirty, &update_area);

    pthread_mutex_unlock(&ssd->lock);

}
