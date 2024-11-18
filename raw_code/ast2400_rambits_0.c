static int ast2400_rambits(AspeedSDMCState *s)

{

    switch (s->ram_size >> 20) {

    case 64:

        return ASPEED_SDMC_DRAM_64MB;

    case 128:

        return ASPEED_SDMC_DRAM_128MB;

    case 256:

        return ASPEED_SDMC_DRAM_256MB;

    case 512:

        return ASPEED_SDMC_DRAM_512MB;

    default:

        break;

    }



    /* use a common default */

    error_report("warning: Invalid RAM size 0x%" PRIx64

                 ". Using default 256M", s->ram_size);

    s->ram_size = 256 << 20;

    return ASPEED_SDMC_DRAM_256MB;

}
