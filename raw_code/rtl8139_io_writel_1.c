static void rtl8139_io_writel(void *opaque, uint8_t addr, uint32_t val)

{

    RTL8139State *s = opaque;



    addr &= 0xfc;



    switch (addr)

    {

        case RxMissed:

            DPRINTF("RxMissed clearing on write\n");

            s->RxMissed = 0;

            break;



        case TxConfig:

            rtl8139_TxConfig_write(s, val);

            break;



        case RxConfig:

            rtl8139_RxConfig_write(s, val);

            break;



        case TxStatus0 ... TxStatus0+4*4-1:

            rtl8139_TxStatus_write(s, addr-TxStatus0, val);

            break;



        case TxAddr0 ... TxAddr0+4*4-1:

            rtl8139_TxAddr_write(s, addr-TxAddr0, val);

            break;



        case RxBuf:

            rtl8139_RxBuf_write(s, val);

            break;



        case RxRingAddrLO:

            DPRINTF("C+ RxRing low bits write val=0x%08x\n", val);

            s->RxRingAddrLO = val;

            break;



        case RxRingAddrHI:

            DPRINTF("C+ RxRing high bits write val=0x%08x\n", val);

            s->RxRingAddrHI = val;

            break;



        case Timer:

            DPRINTF("TCTR Timer reset on write\n");

            s->TCTR_base = qemu_get_clock_ns(vm_clock);

            rtl8139_set_next_tctr_time(s, s->TCTR_base);

            break;



        case FlashReg:

            DPRINTF("FlashReg TimerInt write val=0x%08x\n", val);

            if (s->TimerInt != val) {

                s->TimerInt = val;

                rtl8139_set_next_tctr_time(s, qemu_get_clock_ns(vm_clock));

            }

            break;



        default:

            DPRINTF("ioport write(l) addr=0x%x val=0x%08x via write(b)\n",

                addr, val);

            rtl8139_io_writeb(opaque, addr, val & 0xff);

            rtl8139_io_writeb(opaque, addr + 1, (val >> 8) & 0xff);

            rtl8139_io_writeb(opaque, addr + 2, (val >> 16) & 0xff);

            rtl8139_io_writeb(opaque, addr + 3, (val >> 24) & 0xff);

            break;

    }

}
