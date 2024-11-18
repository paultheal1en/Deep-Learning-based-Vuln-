static void qxl_enter_vga_mode(PCIQXLDevice *d)

{

    if (d->mode == QXL_MODE_VGA) {

        return;

    }

    trace_qxl_enter_vga_mode(d->id);

#if SPICE_SERVER_VERSION >= 0x000c03 /* release 0.12.3 */

    spice_qxl_driver_unload(&d->ssd.qxl);

#endif

    graphic_console_set_hwops(d->ssd.dcl.con, d->vga.hw_ops, &d->vga);

    update_displaychangelistener(&d->ssd.dcl, GUI_REFRESH_INTERVAL_DEFAULT);

    qemu_spice_create_host_primary(&d->ssd);

    d->mode = QXL_MODE_VGA;


    vga_dirty_log_start(&d->vga);

    graphic_hw_update(d->vga.con);

}