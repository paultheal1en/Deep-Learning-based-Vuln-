static inline uint32_t ne2000_mem_readl(NE2000State *s, uint32_t addr)

{

    addr &= ~1; /* XXX: check exact behaviour if not even */

    if (addr < 32 ||

        (addr >= NE2000_PMEM_START && addr < NE2000_MEM_SIZE)) {

        return ldl_le_p(s->mem + addr);

    } else {

        return 0xffffffff;

    }

}
