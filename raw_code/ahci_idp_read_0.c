static uint64_t ahci_idp_read(void *opaque, target_phys_addr_t addr,

                              unsigned size)

{

    AHCIState *s = opaque;



    if (addr == s->idp_offset) {

        /* index register */

        return s->idp_index;

    } else if (addr == s->idp_offset + 4) {

        /* data register - do memory read at location selected by index */

        return ahci_mem_read(opaque, s->idp_index, size);

    } else {

        return 0;

    }

}
