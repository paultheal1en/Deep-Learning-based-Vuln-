nand_write(void *opaque, target_phys_addr_t addr, uint64_t value,

           unsigned size)

{

    struct nand_state_t *s = opaque;

    int rdy;



    DNAND(printf("%s addr=%x v=%x\n", __func__, addr, (unsigned)value));

    nand_setpins(s->nand, s->cle, s->ale, s->ce, 1, 0);

    nand_setio(s->nand, value);

    nand_getpins(s->nand, &rdy);

    s->rdy = rdy;

}
