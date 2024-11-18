static CharDriverState *qemu_chr_open_mux(CharDriverState *drv)

{

    CharDriverState *chr;

    MuxDriver *d;



    chr = g_malloc0(sizeof(CharDriverState));

    d = g_malloc0(sizeof(MuxDriver));



    chr->opaque = d;

    d->drv = drv;

    d->focus = -1;

    chr->chr_write = mux_chr_write;

    chr->chr_update_read_handler = mux_chr_update_read_handler;

    chr->chr_accept_input = mux_chr_accept_input;

    /* Frontend guest-open / -close notification is not support with muxes */

    chr->chr_set_fe_open = NULL;

    /* only default to opened state if we've realized the initial

     * set of muxes

     */

    chr->explicit_be_open = muxes_realized ? 0 : 1;

    chr->is_mux = 1;



    return chr;

}
