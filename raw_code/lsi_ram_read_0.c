static uint64_t lsi_ram_read(void *opaque, target_phys_addr_t addr,

                             unsigned size)

{

    LSIState *s = opaque;

    uint32_t val;

    uint32_t mask;



    val = s->script_ram[addr >> 2];

    mask = ((uint64_t)1 << (size * 8)) - 1;

    val >>= (addr & 3) * 8;

    return val & mask;

}
