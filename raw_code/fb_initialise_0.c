static int fb_initialise(struct XenDevice *xendev)

{

    struct XenFB *fb = container_of(xendev, struct XenFB, c.xendev);

    struct xenfb_page *fb_page;

    int videoram;

    int rc;



    if (xenstore_read_fe_int(xendev, "videoram", &videoram) == -1)

	videoram = 0;



    rc = common_bind(&fb->c);

    if (rc != 0)

	return rc;



    fb_page = fb->c.page;

    rc = xenfb_configure_fb(fb, videoram * 1024 * 1024U,

			    fb_page->width, fb_page->height, fb_page->depth,

			    fb_page->mem_length, 0, fb_page->line_length);

    if (rc != 0)

	return rc;



    rc = xenfb_map_fb(fb);

    if (rc != 0)

	return rc;



#if 0  /* handled in xen_init_display() for now */

    if (!fb->have_console) {

        fb->c.ds = graphic_console_init(xenfb_update,

                                        xenfb_invalidate,

                                        NULL,

                                        NULL,

                                        fb);

        fb->have_console = 1;

    }

#endif



    if (xenstore_read_fe_int(xendev, "feature-update", &fb->feature_update) == -1)

	fb->feature_update = 0;

    if (fb->feature_update)

	xenstore_write_be_int(xendev, "request-update", 1);



    xen_pv_printf(xendev, 1, "feature-update=%d, videoram=%d\n",

		  fb->feature_update, videoram);

    return 0;

}
