static int max7310_tx(I2CSlave *i2c, uint8_t data)

{

    MAX7310State *s = MAX7310(i2c);

    uint8_t diff;

    int line;



    if (s->len ++ > 1) {

#ifdef VERBOSE

        printf("%s: message too long (%i bytes)\n", __FUNCTION__, s->len);

#endif

        return 1;

    }



    if (s->i2c_command_byte) {

        s->command = data;

        s->i2c_command_byte = 0;

        return 0;

    }



    switch (s->command) {

    case 0x01:	/* Output port */

        for (diff = (data ^ s->level) & ~s->direction; diff;

                        diff &= ~(1 << line)) {

            line = ffs(diff) - 1;

            if (s->handler[line])

                qemu_set_irq(s->handler[line], (data >> line) & 1);

        }

        s->level = (s->level & s->direction) | (data & ~s->direction);

        break;



    case 0x02:	/* Polarity inversion */

        s->polarity = data;

        break;



    case 0x03:	/* Configuration */

        s->level &= ~(s->direction ^ data);

        s->direction = data;

        break;



    case 0x04:	/* Timeout */

        s->status = data;

        break;



    case 0x00:	/* Input port - ignore writes */

	break;

    default:

#ifdef VERBOSE

        printf("%s: unknown register %02x\n", __FUNCTION__, s->command);

#endif

        return 1;

    }



    return 0;

}
