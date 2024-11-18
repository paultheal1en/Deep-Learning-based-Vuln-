static uint32_t rtl8139_io_readw(void *opaque, uint8_t addr)

{

    RTL8139State *s = opaque;

    uint32_t ret;



    switch (addr)

    {

        case TxAddr0 ... TxAddr0+4*4-1:

            ret = rtl8139_TxStatus_read(s, addr, 2);

            break;

        case IntrMask:

            ret = rtl8139_IntrMask_read(s);

            break;



        case IntrStatus:

            ret = rtl8139_IntrStatus_read(s);

            break;



        case MultiIntr:

            ret = rtl8139_MultiIntr_read(s);

            break;



        case RxBufPtr:

            ret = rtl8139_RxBufPtr_read(s);

            break;



        case RxBufAddr:

            ret = rtl8139_RxBufAddr_read(s);

            break;



        case BasicModeCtrl:

            ret = rtl8139_BasicModeCtrl_read(s);

            break;

        case BasicModeStatus:

            ret = rtl8139_BasicModeStatus_read(s);

            break;

        case NWayAdvert:

            ret = s->NWayAdvert;

            DPRINTF("NWayAdvert read(w) val=0x%04x\n", ret);

            break;

        case NWayLPAR:

            ret = s->NWayLPAR;

            DPRINTF("NWayLPAR read(w) val=0x%04x\n", ret);

            break;

        case NWayExpansion:

            ret = s->NWayExpansion;

            DPRINTF("NWayExpansion read(w) val=0x%04x\n", ret);

            break;



        case CpCmd:

            ret = rtl8139_CpCmd_read(s);

            break;



        case IntrMitigate:

            ret = rtl8139_IntrMitigate_read(s);

            break;



        case TxSummary:

            ret = rtl8139_TSAD_read(s);

            break;



        case CSCR:

            ret = rtl8139_CSCR_read(s);

            break;



        default:

            DPRINTF("ioport read(w) addr=0x%x via read(b)\n", addr);



            ret  = rtl8139_io_readb(opaque, addr);

            ret |= rtl8139_io_readb(opaque, addr + 1) << 8;



            DPRINTF("ioport read(w) addr=0x%x val=0x%04x\n", addr, ret);

            break;

    }



    return ret;

}
