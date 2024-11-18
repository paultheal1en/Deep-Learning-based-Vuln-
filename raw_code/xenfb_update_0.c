static void xenfb_update(void *opaque)

{

    struct XenFB *xenfb = opaque;

    DisplaySurface *surface;

    int i;



    if (xenfb->c.xendev.be_state != XenbusStateConnected)

        return;



    if (!xenfb->feature_update) {

        /* we don't get update notifications, thus use the

         * sledge hammer approach ... */

        xenfb->up_fullscreen = 1;

    }



    /* resize if needed */

    if (xenfb->do_resize) {

        pixman_format_code_t format;



        xenfb->do_resize = 0;

        switch (xenfb->depth) {

        case 16:

        case 32:

            /* console.c supported depth -> buffer can be used directly */

            format = qemu_default_pixman_format(xenfb->depth, true);

            surface = qemu_create_displaysurface_from

                (xenfb->width, xenfb->height, format,

                 xenfb->row_stride, xenfb->pixels + xenfb->offset);

            break;

        default:

            /* we must convert stuff */

            surface = qemu_create_displaysurface(xenfb->width, xenfb->height);

            break;

        }

        dpy_gfx_replace_surface(xenfb->c.con, surface);

        xen_pv_printf(&xenfb->c.xendev, 1,

                      "update: resizing: %dx%d @ %d bpp%s\n",

                      xenfb->width, xenfb->height, xenfb->depth,

                      is_buffer_shared(surface) ? " (shared)" : "");

        xenfb->up_fullscreen = 1;

    }



    /* run queued updates */

    if (xenfb->up_fullscreen) {

        xen_pv_printf(&xenfb->c.xendev, 3, "update: fullscreen\n");

        xenfb_guest_copy(xenfb, 0, 0, xenfb->width, xenfb->height);

    } else if (xenfb->up_count) {

        xen_pv_printf(&xenfb->c.xendev, 3, "update: %d rects\n",

                      xenfb->up_count);

        for (i = 0; i < xenfb->up_count; i++)

            xenfb_guest_copy(xenfb,

                             xenfb->up_rects[i].x,

                             xenfb->up_rects[i].y,

                             xenfb->up_rects[i].w,

                             xenfb->up_rects[i].h);

    } else {

        xen_pv_printf(&xenfb->c.xendev, 3, "update: nothing\n");

    }

    xenfb->up_count = 0;

    xenfb->up_fullscreen = 0;

}
