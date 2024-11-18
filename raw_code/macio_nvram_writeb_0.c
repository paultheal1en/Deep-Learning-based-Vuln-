static void macio_nvram_writeb(void *opaque, target_phys_addr_t addr,

                               uint64_t value, unsigned size)

{

    MacIONVRAMState *s = opaque;



    addr = (addr >> s->it_shift) & (s->size - 1);

    s->data[addr] = value;

    NVR_DPRINTF("writeb addr %04x val %x\n", (int)addr, value);

}
