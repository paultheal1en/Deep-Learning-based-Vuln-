static int ssd0303_init(I2CSlave *i2c)

{

    ssd0303_state *s = FROM_I2C_SLAVE(ssd0303_state, i2c);



    s->con = graphic_console_init(ssd0303_update_display,

                                  ssd0303_invalidate_display,

                                  NULL, NULL, s);

    qemu_console_resize(s->con, 96 * MAGNIFY, 16 * MAGNIFY);

    return 0;

}
