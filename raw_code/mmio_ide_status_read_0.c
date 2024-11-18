static uint64_t mmio_ide_status_read(void *opaque, target_phys_addr_t addr,

                                     unsigned size)

{

    MMIOState *s= opaque;

    return ide_status_read(&s->bus, 0);

}
