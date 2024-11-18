static uint64_t pl050_read(void *opaque, hwaddr offset,

                           unsigned size)

{

    pl050_state *s = (pl050_state *)opaque;

    if (offset >= 0xfe0 && offset < 0x1000)

        return pl050_id[(offset - 0xfe0) >> 2];



    switch (offset >> 2) {

    case 0: /* KMICR */

        return s->cr;

    case 1: /* KMISTAT */

        {

            uint8_t val;

            uint32_t stat;



            val = s->last;

            val = val ^ (val >> 4);

            val = val ^ (val >> 2);

            val = (val ^ (val >> 1)) & 1;



            stat = PL050_TXEMPTY;

            if (val)

                stat |= PL050_RXPARITY;

            if (s->pending)

                stat |= PL050_RXFULL;



            return stat;

        }

    case 2: /* KMIDATA */

        if (s->pending)

            s->last = ps2_read_data(s->dev);

        return s->last;

    case 3: /* KMICLKDIV */

        return s->clk;

    case 4: /* KMIIR */

        return s->pending | 2;

    default:

        hw_error("pl050_read: Bad offset %x\n", (int)offset);

        return 0;

    }

}
