static uint64_t ehci_opreg_read(void *ptr, hwaddr addr,

                                unsigned size)

{

    EHCIState *s = ptr;

    uint32_t val;



    val = s->opreg[addr >> 2];

    trace_usb_ehci_opreg_read(addr + s->opregbase, addr2str(addr), val);

    return val;

}
