static void ecc_diag_mem_write(void *opaque, target_phys_addr_t addr,

                               uint64_t val, unsigned size)

{

    ECCState *s = opaque;



    trace_ecc_diag_mem_writeb(addr, val);

    s->diag[addr & ECC_DIAG_MASK] = val;

}
