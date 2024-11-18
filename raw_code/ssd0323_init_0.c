static int ssd0323_init(SSISlave *dev)

{

    ssd0323_state *s = FROM_SSI_SLAVE(ssd0323_state, dev);



    s->col_end = 63;

    s->row_end = 79;

    s->con = graphic_console_init(ssd0323_update_display,

                                  ssd0323_invalidate_display,

                                  NULL, NULL, s);

    qemu_console_resize(s->con, 128 * MAGNIFY, 64 * MAGNIFY);



    qdev_init_gpio_in(&dev->qdev, ssd0323_cd, 1);



    register_savevm(&dev->qdev, "ssd0323_oled", -1, 1,

                    ssd0323_save, ssd0323_load, s);

    return 0;

}
