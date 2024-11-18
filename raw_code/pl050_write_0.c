static void pl050_write(void *opaque, hwaddr offset,

                        uint64_t value, unsigned size)

{

    pl050_state *s = (pl050_state *)opaque;

    switch (offset >> 2) {

    case 0: /* KMICR */

        s->cr = value;

        pl050_update(s, s->pending);

        /* ??? Need to implement the enable/disable bit.  */

        break;

    case 2: /* KMIDATA */

        /* ??? This should toggle the TX interrupt line.  */

        /* ??? This means kbd/mouse can block each other.  */

        if (s->is_mouse) {

            ps2_write_mouse(s->dev, value);

        } else {

            ps2_write_keyboard(s->dev, value);

        }

        break;

    case 3: /* KMICLKDIV */

        s->clk = value;

        return;

    default:

        hw_error("pl050_write: Bad offset %x\n", (int)offset);

    }

}
