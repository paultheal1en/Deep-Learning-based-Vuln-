static int interface_client_monitors_config(QXLInstance *sin,

                                        VDAgentMonitorsConfig *monitors_config)

{

    PCIQXLDevice *qxl = container_of(sin, PCIQXLDevice, ssd.qxl);

    QXLRom *rom = memory_region_get_ram_ptr(&qxl->rom_bar);

    int i;

    unsigned max_outputs = ARRAY_SIZE(rom->client_monitors_config.heads);



    if (qxl->revision < 4) {

        trace_qxl_client_monitors_config_unsupported_by_device(qxl->id,

                                                               qxl->revision);

        return 0;

    }

    /*

     * Older windows drivers set int_mask to 0 when their ISR is called,

     * then later set it to ~0. So it doesn't relate to the actual interrupts

     * handled. However, they are old, so clearly they don't support this

     * interrupt

     */

    if (qxl->ram->int_mask == 0 || qxl->ram->int_mask == ~0 ||

        !(qxl->ram->int_mask & QXL_INTERRUPT_CLIENT_MONITORS_CONFIG)) {

        trace_qxl_client_monitors_config_unsupported_by_guest(qxl->id,

                                                            qxl->ram->int_mask,

                                                            monitors_config);

        return 0;

    }

    if (!monitors_config) {

        return 1;

    }



#if SPICE_SERVER_VERSION >= 0x000c06 /* release 0.12.6 */

    /* limit number of outputs based on setting limit */

    if (qxl->max_outputs && qxl->max_outputs <= max_outputs) {

        max_outputs = qxl->max_outputs;

    }

#endif



    memset(&rom->client_monitors_config, 0,

           sizeof(rom->client_monitors_config));

    rom->client_monitors_config.count = monitors_config->num_of_monitors;

    /* monitors_config->flags ignored */

    if (rom->client_monitors_config.count >= max_outputs) {

        trace_qxl_client_monitors_config_capped(qxl->id,

                                monitors_config->num_of_monitors,

                                max_outputs);

        rom->client_monitors_config.count = max_outputs;

    }

    for (i = 0 ; i < rom->client_monitors_config.count ; ++i) {

        VDAgentMonConfig *monitor = &monitors_config->monitors[i];

        QXLURect *rect = &rom->client_monitors_config.heads[i];

        /* monitor->depth ignored */

        rect->left = monitor->x;

        rect->top = monitor->y;

        rect->right = monitor->x + monitor->width;

        rect->bottom = monitor->y + monitor->height;

    }

    rom->client_monitors_config_crc = qxl_crc32(

            (const uint8_t *)&rom->client_monitors_config,

            sizeof(rom->client_monitors_config));

    trace_qxl_client_monitors_config_crc(qxl->id,

            sizeof(rom->client_monitors_config),

            rom->client_monitors_config_crc);



    trace_qxl_interrupt_client_monitors_config(qxl->id,

                        rom->client_monitors_config.count,

                        rom->client_monitors_config.heads);

    qxl_send_events(qxl, QXL_INTERRUPT_CLIENT_MONITORS_CONFIG);

    return 1;

}
