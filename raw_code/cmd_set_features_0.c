static bool cmd_set_features(IDEState *s, uint8_t cmd)

{

    uint16_t *identify_data;



    if (!s->bs) {

        ide_abort_command(s);

        return true;

    }



    /* XXX: valid for CDROM ? */

    switch (s->feature) {

    case 0x02: /* write cache enable */

        bdrv_set_enable_write_cache(s->bs, true);

        identify_data = (uint16_t *)s->identify_data;

        put_le16(identify_data + 85, (1 << 14) | (1 << 5) | 1);

        return true;

    case 0x82: /* write cache disable */

        bdrv_set_enable_write_cache(s->bs, false);

        identify_data = (uint16_t *)s->identify_data;

        put_le16(identify_data + 85, (1 << 14) | 1);

        ide_flush_cache(s);

        return false;

    case 0xcc: /* reverting to power-on defaults enable */

    case 0x66: /* reverting to power-on defaults disable */

    case 0xaa: /* read look-ahead enable */

    case 0x55: /* read look-ahead disable */

    case 0x05: /* set advanced power management mode */

    case 0x85: /* disable advanced power management mode */

    case 0x69: /* NOP */

    case 0x67: /* NOP */

    case 0x96: /* NOP */

    case 0x9a: /* NOP */

    case 0x42: /* enable Automatic Acoustic Mode */

    case 0xc2: /* disable Automatic Acoustic Mode */

        return true;

    case 0x03: /* set transfer mode */

        {

            uint8_t val = s->nsector & 0x07;

            identify_data = (uint16_t *)s->identify_data;



            switch (s->nsector >> 3) {

            case 0x00: /* pio default */

            case 0x01: /* pio mode */

                put_le16(identify_data + 62, 0x07);

                put_le16(identify_data + 63, 0x07);

                put_le16(identify_data + 88, 0x3f);

                break;

            case 0x02: /* sigle word dma mode*/

                put_le16(identify_data + 62, 0x07 | (1 << (val + 8)));

                put_le16(identify_data + 63, 0x07);

                put_le16(identify_data + 88, 0x3f);

                break;

            case 0x04: /* mdma mode */

                put_le16(identify_data + 62, 0x07);

                put_le16(identify_data + 63, 0x07 | (1 << (val + 8)));

                put_le16(identify_data + 88, 0x3f);

                break;

            case 0x08: /* udma mode */

                put_le16(identify_data + 62, 0x07);

                put_le16(identify_data + 63, 0x07);

                put_le16(identify_data + 88, 0x3f | (1 << (val + 8)));

                break;

            default:

                goto abort_cmd;

            }

            return true;

        }

    }



abort_cmd:

    ide_abort_command(s);

    return true;

}
