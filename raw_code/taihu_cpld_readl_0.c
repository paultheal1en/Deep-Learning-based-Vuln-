static uint32_t taihu_cpld_readl (void *opaque, hwaddr addr)

{

    uint32_t ret;



    ret = taihu_cpld_readb(opaque, addr) << 24;

    ret |= taihu_cpld_readb(opaque, addr + 1) << 16;

    ret |= taihu_cpld_readb(opaque, addr + 2) << 8;

    ret |= taihu_cpld_readb(opaque, addr + 3);



    return ret;

}
