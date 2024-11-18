static void macio_nvram_writeb(void *opaque, hwaddr addr,

                               uint64_t value, unsigned size)

{

    MacIONVRAMState *s = opaque;



    addr = (addr >> s->it_shift) & (s->size - 1);

    s->data[addr] = value;

    NVR_DPRINTF("writeb addr %04" PHYS_PRIx " val %" PRIx64 "\n", addr, value);

}
