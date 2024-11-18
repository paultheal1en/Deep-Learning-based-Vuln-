static void omap_rfbi_transfer_start(struct omap_dss_s *s)

{

    void *data;

    hwaddr len;

    hwaddr data_addr;

    int pitch;

    static void *bounce_buffer;

    static hwaddr bounce_len;



    if (!s->rfbi.enable || s->rfbi.busy)

        return;



    if (s->rfbi.control & (1 << 1)) {				/* BYPASS */

        /* TODO: in non-Bypass mode we probably need to just assert the

         * DRQ and wait for DMA to write the pixels.  */

        fprintf(stderr, "%s: Bypass mode unimplemented\n", __FUNCTION__);

        return;

    }



    if (!(s->dispc.control & (1 << 11)))			/* RFBIMODE */

        return;

    /* TODO: check that LCD output is enabled in DISPC.  */



    s->rfbi.busy = 1;



    len = s->rfbi.pixels * 2;



    data_addr = s->dispc.l[0].addr[0];

    data = cpu_physical_memory_map(data_addr, &len, 0);

    if (data && len != s->rfbi.pixels * 2) {

        cpu_physical_memory_unmap(data, len, 0, 0);

        data = NULL;

        len = s->rfbi.pixels * 2;

    }

    if (!data) {

        if (len > bounce_len) {

            bounce_buffer = g_realloc(bounce_buffer, len);

        }

        data = bounce_buffer;

        cpu_physical_memory_read(data_addr, data, len);

    }



    /* TODO bpp */

    s->rfbi.pixels = 0;



    /* TODO: negative values */

    pitch = s->dispc.l[0].nx + (s->dispc.l[0].rowinc - 1) / 2;



    if ((s->rfbi.control & (1 << 2)) && s->rfbi.chip[0])

        s->rfbi.chip[0]->block(s->rfbi.chip[0]->opaque, 1, data, len, pitch);

    if ((s->rfbi.control & (1 << 3)) && s->rfbi.chip[1])

        s->rfbi.chip[1]->block(s->rfbi.chip[1]->opaque, 1, data, len, pitch);



    if (data != bounce_buffer) {

        cpu_physical_memory_unmap(data, len, 0, len);

    }



    omap_rfbi_transfer_stop(s);



    /* TODO */

    s->dispc.irqst |= 1;					/* FRAMEDONE */

    omap_dispc_interrupt_update(s);

}
