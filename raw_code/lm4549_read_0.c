uint32_t lm4549_read(lm4549_state *s, target_phys_addr_t offset)

{

    uint16_t *regfile = s->regfile;

    uint32_t value = 0;



    /* Read the stored value */

    assert(offset < 128);

    value = regfile[offset];



    DPRINTF("read [0x%02x] = 0x%04x\n", offset, value);



    return value;

}
