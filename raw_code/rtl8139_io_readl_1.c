static uint32_t rtl8139_io_readl(void *opaque, uint8_t addr)

{

    RTL8139State *s = opaque;

    uint32_t ret;



    switch (addr)

    {

        case RxMissed:

            ret = s->RxMissed;



            DPRINTF("RxMissed read val=0x%08x\n", ret);

            break;



        case TxConfig:

            ret = rtl8139_TxConfig_read(s);

            break;



        case RxConfig:

            ret = rtl8139_RxConfig_read(s);

            break;



        case TxStatus0 ... TxStatus0+4*4-1:

            ret = rtl8139_TxStatus_read(s, addr, 4);

            break;



        case TxAddr0 ... TxAddr0+4*4-1:

            ret = rtl8139_TxAddr_read(s, addr-TxAddr0);

            break;



        case RxBuf:

            ret = rtl8139_RxBuf_read(s);

            break;



        case RxRingAddrLO:

            ret = s->RxRingAddrLO;

            DPRINTF("C+ RxRing low bits read val=0x%08x\n", ret);

            break;



        case RxRingAddrHI:

            ret = s->RxRingAddrHI;

            DPRINTF("C+ RxRing high bits read val=0x%08x\n", ret);

            break;



        case Timer:

            ret = muldiv64(qemu_get_clock_ns(vm_clock) - s->TCTR_base,

                           PCI_FREQUENCY, get_ticks_per_sec());

            DPRINTF("TCTR Timer read val=0x%08x\n", ret);

            break;



        case FlashReg:

            ret = s->TimerInt;

            DPRINTF("FlashReg TimerInt read val=0x%08x\n", ret);

            break;



        default:

            DPRINTF("ioport read(l) addr=0x%x via read(b)\n", addr);



            ret  = rtl8139_io_readb(opaque, addr);

            ret |= rtl8139_io_readb(opaque, addr + 1) << 8;

            ret |= rtl8139_io_readb(opaque, addr + 2) << 16;

            ret |= rtl8139_io_readb(opaque, addr + 3) << 24;



            DPRINTF("read(l) addr=0x%x val=%08x\n", addr, ret);

            break;

    }



    return ret;

}
