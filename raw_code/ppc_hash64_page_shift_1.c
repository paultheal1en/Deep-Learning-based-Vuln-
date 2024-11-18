static uint64_t ppc_hash64_page_shift(ppc_slb_t *slb)

{

    uint64_t epnshift;



    /* Page size according to the SLB, which we use to generate the

     * EPN for hash table lookup..  When we implement more recent MMU

     * extensions this might be different from the actual page size

     * encoded in the PTE */

    if ((slb->vsid & SLB_VSID_LLP_MASK) == SLB_VSID_4K) {

        epnshift = TARGET_PAGE_BITS;

    } else if ((slb->vsid & SLB_VSID_LLP_MASK) == SLB_VSID_64K) {

        epnshift = TARGET_PAGE_BITS_64K;

    } else {

        epnshift = TARGET_PAGE_BITS_16M;

    }

    return epnshift;

}
