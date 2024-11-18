void helper_ldqf(CPUSPARCState *env, target_ulong addr, int mem_idx)

{

    /* XXX add 128 bit load */

    CPU_QuadU u;



    helper_check_align(env, addr, 7);

#if !defined(CONFIG_USER_ONLY)

    switch (mem_idx) {

    case MMU_USER_IDX:

        u.ll.upper = cpu_ldq_user(env, addr);

        u.ll.lower = cpu_ldq_user(env, addr + 8);

        QT0 = u.q;

        break;

    case MMU_KERNEL_IDX:

        u.ll.upper = cpu_ldq_kernel(env, addr);

        u.ll.lower = cpu_ldq_kernel(env, addr + 8);

        QT0 = u.q;

        break;

#ifdef TARGET_SPARC64

    case MMU_HYPV_IDX:

        u.ll.upper = cpu_ldq_hypv(env, addr);

        u.ll.lower = cpu_ldq_hypv(env, addr + 8);

        QT0 = u.q;

        break;

#endif

    default:

        DPRINTF_MMU("helper_ldqf: need to check MMU idx %d\n", mem_idx);

        break;

    }

#else

    u.ll.upper = ldq_raw(address_mask(env, addr));

    u.ll.lower = ldq_raw(address_mask(env, addr + 8));

    QT0 = u.q;

#endif

}
