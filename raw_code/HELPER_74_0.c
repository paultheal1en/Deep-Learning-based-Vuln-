void HELPER(idte)(CPUS390XState *env, uint64_t r1, uint64_t r2, uint32_t m4)

{

    CPUState *cs = CPU(s390_env_get_cpu(env));

    const uintptr_t ra = GETPC();

    uint64_t table, entry, raddr;

    uint16_t entries, i, index = 0;



    if (r2 & 0xff000) {

        cpu_restore_state(cs, ra);

        program_interrupt(env, PGM_SPECIFICATION, 4);

    }



    if (!(r2 & 0x800)) {

        /* invalidation-and-clearing operation */

        table = r1 & _ASCE_ORIGIN;

        entries = (r2 & 0x7ff) + 1;



        switch (r1 & _ASCE_TYPE_MASK) {

        case _ASCE_TYPE_REGION1:

            index = (r2 >> 53) & 0x7ff;

            break;

        case _ASCE_TYPE_REGION2:

            index = (r2 >> 42) & 0x7ff;

            break;

        case _ASCE_TYPE_REGION3:

            index = (r2 >> 31) & 0x7ff;

            break;

        case _ASCE_TYPE_SEGMENT:

            index = (r2 >> 20) & 0x7ff;

            break;

        }

        for (i = 0; i < entries; i++) {

            /* addresses are not wrapped in 24/31bit mode but table index is */

            raddr = table + ((index + i) & 0x7ff) * sizeof(entry);

            entry = cpu_ldq_real_ra(env, raddr, ra);

            if (!(entry & _REGION_ENTRY_INV)) {

                /* we are allowed to not store if already invalid */

                entry |= _REGION_ENTRY_INV;

                cpu_stq_real_ra(env, raddr, entry, ra);

            }

        }

    }



    /* We simply flush the complete tlb, therefore we can ignore r3. */

    if (m4 & 1) {

        tlb_flush(cs);

    } else {

        tlb_flush_all_cpus_synced(cs);

    }

}
