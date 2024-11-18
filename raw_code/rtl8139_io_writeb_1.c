static void rtl8139_io_writeb(void *opaque, uint8_t addr, uint32_t val)

{

    RTL8139State *s = opaque;



    addr &= 0xff;



    switch (addr)

    {

        case MAC0 ... MAC0+5:

            s->phys[addr - MAC0] = val;

            break;

        case MAC0+6 ... MAC0+7:

            /* reserved */

            break;

        case MAR0 ... MAR0+7:

            s->mult[addr - MAR0] = val;

            break;

        case ChipCmd:

            rtl8139_ChipCmd_write(s, val);

            break;

        case Cfg9346:

            rtl8139_Cfg9346_write(s, val);

            break;

        case TxConfig: /* windows driver sometimes writes using byte-lenth call */

            rtl8139_TxConfig_writeb(s, val);

            break;

        case Config0:

            rtl8139_Config0_write(s, val);

            break;

        case Config1:

            rtl8139_Config1_write(s, val);

            break;

        case Config3:

            rtl8139_Config3_write(s, val);

            break;

        case Config4:

            rtl8139_Config4_write(s, val);

            break;

        case Config5:

            rtl8139_Config5_write(s, val);

            break;

        case MediaStatus:

            /* ignore */

            DPRINTF("not implemented write(b) to MediaStatus val=0x%02x\n",

                val);

            break;



        case HltClk:

            DPRINTF("HltClk write val=0x%08x\n", val);

            if (val == 'R')

            {

                s->clock_enabled = 1;

            }

            else if (val == 'H')

            {

                s->clock_enabled = 0;

            }

            break;



        case TxThresh:

            DPRINTF("C+ TxThresh write(b) val=0x%02x\n", val);

            s->TxThresh = val;

            break;



        case TxPoll:

            DPRINTF("C+ TxPoll write(b) val=0x%02x\n", val);

            if (val & (1 << 7))

            {

                DPRINTF("C+ TxPoll high priority transmission (not "

                    "implemented)\n");

                //rtl8139_cplus_transmit(s);

            }

            if (val & (1 << 6))

            {

                DPRINTF("C+ TxPoll normal priority transmission\n");

                rtl8139_cplus_transmit(s);

            }



            break;



        default:

            DPRINTF("not implemented write(b) addr=0x%x val=0x%02x\n", addr,

                val);

            break;

    }

}
