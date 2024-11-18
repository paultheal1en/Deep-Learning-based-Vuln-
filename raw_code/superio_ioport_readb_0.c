static uint32_t superio_ioport_readb(void *opaque, uint32_t addr)

{

    SuperIOConfig *superio_conf = opaque;



    DPRINTF("superio_ioport_readb  address 0x%x   \n", addr);

    return (superio_conf->config[superio_conf->index]);

}
