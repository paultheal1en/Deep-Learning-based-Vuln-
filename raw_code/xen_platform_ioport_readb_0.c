static uint32_t xen_platform_ioport_readb(void *opaque, uint32_t addr)

{

    addr &= 0xff;



    if (addr == 0) {

        return platform_fixed_ioport_readb(opaque, XEN_PLATFORM_IOPORT);

    } else {

        return ~0u;

    }

}
