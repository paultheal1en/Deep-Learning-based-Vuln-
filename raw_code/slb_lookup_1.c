static int slb_lookup (CPUState *env, target_ulong eaddr,

                       target_ulong *vsid, target_ulong *page_mask, int *attr)

{

    target_phys_addr_t sr_base;

    target_ulong mask;

    uint64_t tmp64;

    uint32_t tmp;

    int n, ret;

    int slb_nr;



    ret = -5;

    sr_base = env->spr[SPR_ASR];

    mask = 0x0000000000000000ULL; /* Avoid gcc warning */

#if 0 /* XXX: Fix this */

    slb_nr = env->slb_nr;

#else

    slb_nr = 32;

#endif

    for (n = 0; n < slb_nr; n++) {

        tmp64 = ldq_phys(sr_base);

        if (tmp64 & 0x0000000008000000ULL) {

            /* SLB entry is valid */

            switch (tmp64 & 0x0000000006000000ULL) {

            case 0x0000000000000000ULL:

                /* 256 MB segment */

                mask = 0xFFFFFFFFF0000000ULL;

                break;

            case 0x0000000002000000ULL:

                /* 1 TB segment */

                mask = 0xFFFF000000000000ULL;

                break;

            case 0x0000000004000000ULL:

            case 0x0000000006000000ULL:

                /* Reserved => segment is invalid */

                continue;

            }

            if ((eaddr & mask) == (tmp64 & mask)) {

                /* SLB match */

                tmp = ldl_phys(sr_base + 8);

                *vsid = ((tmp64 << 24) | (tmp >> 8)) & 0x0003FFFFFFFFFFFFULL;

                *page_mask = ~mask;

                *attr = tmp & 0xFF;

                ret = 0;

                break;

            }

        }

        sr_base += 12;

    }



    return ret;

}
