void HELPER(stby_e)(CPUHPPAState *env, target_ulong addr, target_ulong val)

{

    uintptr_t ra = GETPC();



    switch (addr & 3) {

    case 3:

        /* The 3 byte store must appear atomic.  */

        if (parallel_cpus) {

            atomic_store_3(env, addr - 3, val, 0xffffff00u, ra);

        } else {

            cpu_stw_data_ra(env, addr - 3, val >> 16, ra);

            cpu_stb_data_ra(env, addr - 1, val >> 8, ra);

        }

        break;

    case 2:

        cpu_stw_data_ra(env, addr - 2, val >> 16, ra);

        break;

    case 1:

        cpu_stb_data_ra(env, addr - 1, val >> 24, ra);

        break;

    default:

        /* Nothing is stored, but protection is checked and the

           cacheline is marked dirty.  */

#ifndef CONFIG_USER_ONLY

        probe_write(env, addr, cpu_mmu_index(env, 0), ra);

#endif

        break;

    }

}
