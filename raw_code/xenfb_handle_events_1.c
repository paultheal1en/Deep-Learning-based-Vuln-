static void xenfb_handle_events(struct XenFB *xenfb)

{

    uint32_t prod, cons;

    struct xenfb_page *page = xenfb->c.page;



    prod = page->out_prod;

    if (prod == page->out_cons)

	return;

    xen_rmb();		/* ensure we see ring contents up to prod */

    for (cons = page->out_cons; cons != prod; cons++) {

	union xenfb_out_event *event = &XENFB_OUT_RING_REF(page, cons);

	int x, y, w, h;



	switch (event->type) {

	case XENFB_TYPE_UPDATE:

	    if (xenfb->up_count == UP_QUEUE)

		xenfb->up_fullscreen = 1;

	    if (xenfb->up_fullscreen)

		break;

	    x = MAX(event->update.x, 0);

	    y = MAX(event->update.y, 0);

	    w = MIN(event->update.width, xenfb->width - x);

	    h = MIN(event->update.height, xenfb->height - y);

	    if (w < 0 || h < 0) {

                xen_be_printf(&xenfb->c.xendev, 1, "bogus update ignored\n");

		break;

	    }

	    if (x != event->update.x ||

                y != event->update.y ||

		w != event->update.width ||

		h != event->update.height) {

                xen_be_printf(&xenfb->c.xendev, 1, "bogus update clipped\n");

	    }

	    if (w == xenfb->width && h > xenfb->height / 2) {

		/* scroll detector: updated more than 50% of the lines,

		 * don't bother keeping track of the rectangles then */

		xenfb->up_fullscreen = 1;

	    } else {

		xenfb->up_rects[xenfb->up_count].x = x;

		xenfb->up_rects[xenfb->up_count].y = y;

		xenfb->up_rects[xenfb->up_count].w = w;

		xenfb->up_rects[xenfb->up_count].h = h;

		xenfb->up_count++;

	    }

	    break;

#ifdef XENFB_TYPE_RESIZE

	case XENFB_TYPE_RESIZE:

	    if (xenfb_configure_fb(xenfb, xenfb->fb_len,

				   event->resize.width,

				   event->resize.height,

				   event->resize.depth,

				   xenfb->fb_len,

				   event->resize.offset,

				   event->resize.stride) < 0)

		break;

	    xenfb_invalidate(xenfb);

	    break;

#endif

	}

    }

    xen_mb();		/* ensure we're done with ring contents */

    page->out_cons = cons;

}
