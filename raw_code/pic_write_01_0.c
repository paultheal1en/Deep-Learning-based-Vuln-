static void pic_write(void *opaque, target_phys_addr_t addr,

                      uint64_t value, unsigned int size)

{

    struct etrax_pic *fs = opaque;

    D(printf("%s addr=%x val=%x\n", __func__, addr, value));



    if (addr == R_RW_MASK) {

        fs->regs[R_RW_MASK] = value;

        pic_update(fs);

    }

}
