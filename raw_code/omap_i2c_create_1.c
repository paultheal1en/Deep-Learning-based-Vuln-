I2CAdapter *omap_i2c_create(uint64_t addr)

{

    OMAPI2C *s = g_malloc0(sizeof(*s));

    I2CAdapter *i2c = (I2CAdapter *)s;

    uint16_t data;



    s->addr = addr;



    i2c->send = omap_i2c_send;

    i2c->recv = omap_i2c_recv;



    /* verify the mmio address by looking for a known signature */

    memread(addr + OMAP_I2C_REV, &data, 2);

    g_assert_cmphex(data, ==, 0x34);



    return i2c;

}
