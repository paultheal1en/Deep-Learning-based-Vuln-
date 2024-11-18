void HELPER(stby_b)(CPUHPPAState *env, target_ulong addr, target_ulong val)

{

    uintptr_t ra = GETPC();



    switch (addr & 3) {

    case 3:

        cpu_stb_data_ra(env, addr, val, ra);

        break;

    case 2:

        cpu_stw_data_ra(env, addr, val, ra);

        break;

    case 1:

        /* The 3 byte store must appear atomic.  */

        if (parallel_cpus) {

            atomic_store_3(env, addr, val, 0x00ffffffu, ra);

        } else {

            cpu_stb_data_ra(env, addr, val >> 16, ra);

            cpu_stw_data_ra(env, addr + 1, val, ra);

        }

        break;

    default:

        cpu_stl_data_ra(env, addr, val, ra);

        break;

    }

}
