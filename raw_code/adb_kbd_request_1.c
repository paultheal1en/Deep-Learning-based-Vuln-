static int adb_kbd_request(ADBDevice *d, uint8_t *obuf,

                           const uint8_t *buf, int len)

{

    KBDState *s = ADB_KEYBOARD(d);

    int cmd, reg, olen;



    if ((buf[0] & 0x0f) == ADB_FLUSH) {

        /* flush keyboard fifo */

        s->wptr = s->rptr = s->count = 0;

        return 0;

    }



    cmd = buf[0] & 0xc;

    reg = buf[0] & 0x3;

    olen = 0;

    switch(cmd) {

    case ADB_WRITEREG:

        switch(reg) {

        case 2:

            /* LED status */

            break;

        case 3:

            switch(buf[2]) {

            case ADB_CMD_SELF_TEST:

                break;

            case ADB_CMD_CHANGE_ID:

            case ADB_CMD_CHANGE_ID_AND_ACT:

            case ADB_CMD_CHANGE_ID_AND_ENABLE:

                d->devaddr = buf[1] & 0xf;

                break;

            default:

                d->devaddr = buf[1] & 0xf;

                /* we support handlers:

                 * 1: Apple Standard Keyboard

                 * 2: Apple Extended Keyboard (LShift = RShift)

                 * 3: Apple Extended Keyboard (LShift != RShift)

                 */

                if (buf[2] == 1 || buf[2] == 2 || buf[2] == 3) {

                    d->handler = buf[2];

                }

                break;

            }

        }

        break;

    case ADB_READREG:

        switch(reg) {

        case 0:

            olen = adb_kbd_poll(d, obuf);

            break;

        case 1:

            break;

        case 2:

            obuf[0] = 0x00; /* XXX: check this */

            obuf[1] = 0x07; /* led status */

            olen = 2;

            break;

        case 3:

            obuf[0] = d->handler;

            obuf[1] = d->devaddr;

            olen = 2;

            break;

        }

        break;

    }

    return olen;

}
