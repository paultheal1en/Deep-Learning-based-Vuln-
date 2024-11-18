static uint32_t ppc_hash64_pte_size_decode(uint64_t pte1, uint32_t slb_pshift)

{

    switch (slb_pshift) {

    case 12:

        return 12;

    case 16:

        if ((pte1 & 0xf000) == 0x1000) {

            return 16;

        }

        return 0;

    case 24:

        if ((pte1 & 0xff000) == 0) {

            return 24;

        }

        return 0;

    }

    return 0;

}
