static void omap_i2c_recv(I2CAdapter *i2c, uint8_t addr,

                          uint8_t *buf, uint16_t len)

{

    OMAPI2C *s = (OMAPI2C *)i2c;

    uint16_t data, stat;



    omap_i2c_set_slave_addr(s, addr);



    data = len;

    memwrite(s->addr + OMAP_I2C_CNT, &data, 2);



    data = OMAP_I2C_CON_I2C_EN |

           OMAP_I2C_CON_MST |

           OMAP_I2C_CON_STT |

           OMAP_I2C_CON_STP;

    memwrite(s->addr + OMAP_I2C_CON, &data, 2);

    memread(s->addr + OMAP_I2C_CON, &data, 2);

    g_assert((data & OMAP_I2C_CON_STP) == 0);



    memread(s->addr + OMAP_I2C_STAT, &data, 2);

    g_assert((data & OMAP_I2C_STAT_NACK) == 0);



    memread(s->addr + OMAP_I2C_CNT, &data, 2);

    g_assert_cmpuint(data, ==, len);



    while (len > 0) {

        memread(s->addr + OMAP_I2C_STAT, &data, 2);

        g_assert((data & OMAP_I2C_STAT_RRDY) != 0);

        g_assert((data & OMAP_I2C_STAT_ROVR) == 0);



        memread(s->addr + OMAP_I2C_DATA, &data, 2);



        memread(s->addr + OMAP_I2C_STAT, &stat, 2);

        if (unlikely(len == 1)) {

            *buf = data & 0xf;

            buf++;

            len--;

        } else {

            memcpy(buf, &data, 2);

            buf += 2;

            len -= 2;

        }

    }



    memread(s->addr + OMAP_I2C_CON, &data, 2);

    g_assert((data & OMAP_I2C_CON_STP) == 0);

}
