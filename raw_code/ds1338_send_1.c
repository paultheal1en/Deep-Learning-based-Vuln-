static int ds1338_send(I2CSlave *i2c, uint8_t data)

{

    DS1338State *s = FROM_I2C_SLAVE(DS1338State, i2c);

    if (s->addr_byte) {

        s->ptr = data;

        s->addr_byte = 0;

        return 0;

    }

    s->nvram[s->ptr - 8] = data;

    if (data < 8) {

        qemu_get_timedate(&s->now, s->offset);

        switch(data) {

        case 0:

            /* TODO: Implement CH (stop) bit.  */

            s->now.tm_sec = from_bcd(data & 0x7f);

            break;

        case 1:

            s->now.tm_min = from_bcd(data & 0x7f);

            break;

        case 2:

            if (data & 0x40) {

                if (data & 0x20) {

                    data = from_bcd(data & 0x4f) + 11;

                } else {

                    data = from_bcd(data & 0x1f) - 1;

                }

            } else {

                data = from_bcd(data);

            }

            s->now.tm_hour = data;

            break;

        case 3:

            s->now.tm_wday = from_bcd(data & 7) - 1;

            break;

        case 4:

            s->now.tm_mday = from_bcd(data & 0x3f);

            break;

        case 5:

            s->now.tm_mon = from_bcd(data & 0x1f) - 1;

            break;

        case 6:

            s->now.tm_year = from_bcd(data) + 100;

            break;

        case 7:

            /* Control register. Currently ignored.  */

            break;

        }

        s->offset = qemu_timedate_diff(&s->now);

    }

    s->ptr = (s->ptr + 1) & 0xff;

    return 0;

}
