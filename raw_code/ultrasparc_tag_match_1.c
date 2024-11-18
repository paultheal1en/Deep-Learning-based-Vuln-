static inline int ultrasparc_tag_match(SparcTLBEntry *tlb,

                                       uint64_t address, uint64_t context,

                                       target_phys_addr_t *physical,

                                       int is_nucleus)

{

    uint64_t mask;



    switch ((tlb->tte >> 61) & 3) {

    default:

    case 0x0: // 8k

        mask = 0xffffffffffffe000ULL;

        break;

    case 0x1: // 64k

        mask = 0xffffffffffff0000ULL;

        break;

    case 0x2: // 512k

        mask = 0xfffffffffff80000ULL;

        break;

    case 0x3: // 4M

        mask = 0xffffffffffc00000ULL;

        break;

    }



    // valid, context match, virtual address match?

    if (TTE_IS_VALID(tlb->tte) &&

        ((is_nucleus && compare_masked(0, tlb->tag, 0x1fff))

         || TTE_IS_GLOBAL(tlb->tte) || compare_masked(context, tlb->tag, 0x1fff))

        && compare_masked(address, tlb->tag, mask))

    {

        // decode physical address

        *physical = ((tlb->tte & mask) | (address & ~mask)) & 0x1ffffffe000ULL;

        return 1;

    }



    return 0;

}
