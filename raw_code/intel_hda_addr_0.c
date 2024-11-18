static hwaddr intel_hda_addr(uint32_t lbase, uint32_t ubase)

{

    hwaddr addr;



    addr = ((uint64_t)ubase << 32) | lbase;

    return addr;

}
