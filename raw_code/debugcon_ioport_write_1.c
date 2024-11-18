static void debugcon_ioport_write(void *opaque, hwaddr addr, uint64_t val,

                                  unsigned width)

{

    DebugconState *s = opaque;

    unsigned char ch = val;



#ifdef DEBUG_DEBUGCON

    printf(" [debugcon: write addr=0x%04" HWADDR_PRIx " val=0x%02" PRIx64 "]\n", addr, val);

#endif



    qemu_chr_fe_write(s->chr, &ch, 1);

}
