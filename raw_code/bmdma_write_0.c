static void bmdma_write(void *opaque, target_phys_addr_t addr,

                        uint64_t val, unsigned size)

{

    BMDMAState *bm = opaque;



    if (size != 1) {

        return;

    }



#ifdef DEBUG_IDE

    printf("bmdma: writeb 0x%02x : 0x%02x\n", addr, val);

#endif

    switch (addr & 3) {

    case 0:

        bmdma_cmd_writeb(bm, val);

        break;

    case 2:

        bm->status = (val & 0x60) | (bm->status & 1) | (bm->status & ~val & 0x06);

        break;

    default:;

    }

}
