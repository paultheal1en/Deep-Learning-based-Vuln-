static void do_mac_write(lan9118_state *s, int reg, uint32_t val)

{

    switch (reg) {

    case MAC_CR:

        if ((s->mac_cr & MAC_CR_RXEN) != 0 && (val & MAC_CR_RXEN) == 0) {

            s->int_sts |= RXSTOP_INT;

        }

        s->mac_cr = val & ~MAC_CR_RESERVED;

        DPRINTF("MAC_CR: %08x\n", val);

        break;

    case MAC_ADDRH:

        s->conf.macaddr.a[4] = val & 0xff;

        s->conf.macaddr.a[5] = (val >> 8) & 0xff;

        lan9118_mac_changed(s);

        break;

    case MAC_ADDRL:

        s->conf.macaddr.a[0] = val & 0xff;

        s->conf.macaddr.a[1] = (val >> 8) & 0xff;

        s->conf.macaddr.a[2] = (val >> 16) & 0xff;

        s->conf.macaddr.a[3] = (val >> 24) & 0xff;

        lan9118_mac_changed(s);

        break;

    case MAC_HASHH:

        s->mac_hashh = val;

        break;

    case MAC_HASHL:

        s->mac_hashl = val;

        break;

    case MAC_MII_ACC:

        s->mac_mii_acc = val & 0xffc2;

        if (val & 2) {

            DPRINTF("PHY write %d = 0x%04x\n",

                    (val >> 6) & 0x1f, s->mac_mii_data);

            do_phy_write(s, (val >> 6) & 0x1f, s->mac_mii_data);

        } else {

            s->mac_mii_data = do_phy_read(s, (val >> 6) & 0x1f);

            DPRINTF("PHY read %d = 0x%04x\n",

                    (val >> 6) & 0x1f, s->mac_mii_data);

        }

        break;

    case MAC_MII_DATA:

        s->mac_mii_data = val & 0xffff;

        break;

    case MAC_FLOW:

        s->mac_flow = val & 0xffff0000;

        break;

    case MAC_VLAN1:

        /* Writing to this register changes a condition for

         * FrameTooLong bit in rx_status.  Since we do not set

         * FrameTooLong anyway, just ignore write to this.

         */

        break;

    default:

        hw_error("lan9118: Unimplemented MAC register write: %d = 0x%x\n",

                 s->mac_cmd & 0xf, val);

    }

}
