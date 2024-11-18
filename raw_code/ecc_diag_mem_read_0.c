static uint64_t ecc_diag_mem_read(void *opaque, target_phys_addr_t addr,

                                  unsigned size)

{

    ECCState *s = opaque;

    uint32_t ret = s->diag[(int)addr];



    trace_ecc_diag_mem_readb(addr, ret);

    return ret;

}
