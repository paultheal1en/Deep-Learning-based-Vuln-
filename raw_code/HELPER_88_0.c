void HELPER(cas2l)(CPUM68KState *env, uint32_t regs, uint32_t a1, uint32_t a2)

{

    uint32_t Dc1 = extract32(regs, 9, 3);

    uint32_t Dc2 = extract32(regs, 6, 3);

    uint32_t Du1 = extract32(regs, 3, 3);

    uint32_t Du2 = extract32(regs, 0, 3);

    uint32_t c1 = env->dregs[Dc1];

    uint32_t c2 = env->dregs[Dc2];

    uint32_t u1 = env->dregs[Du1];

    uint32_t u2 = env->dregs[Du2];

    uint32_t l1, l2;

    uintptr_t ra = GETPC();

#if defined(CONFIG_ATOMIC64) && !defined(CONFIG_USER_ONLY)

    int mmu_idx = cpu_mmu_index(env, 0);

    TCGMemOpIdx oi;

#endif



    if (parallel_cpus) {

        /* We're executing in a parallel context -- must be atomic.  */

#ifdef CONFIG_ATOMIC64

        uint64_t c, u, l;

        if ((a1 & 7) == 0 && a2 == a1 + 4) {

            c = deposit64(c2, 32, 32, c1);

            u = deposit64(u2, 32, 32, u1);

#ifdef CONFIG_USER_ONLY

            l = helper_atomic_cmpxchgq_be(env, a1, c, u);

#else

            oi = make_memop_idx(MO_BEQ, mmu_idx);

            l = helper_atomic_cmpxchgq_be_mmu(env, a1, c, u, oi, ra);

#endif

            l1 = l >> 32;

            l2 = l;

        } else if ((a2 & 7) == 0 && a1 == a2 + 4) {

            c = deposit64(c1, 32, 32, c2);

            u = deposit64(u1, 32, 32, u2);

#ifdef CONFIG_USER_ONLY

            l = helper_atomic_cmpxchgq_be(env, a2, c, u);

#else

            oi = make_memop_idx(MO_BEQ, mmu_idx);

            l = helper_atomic_cmpxchgq_be_mmu(env, a2, c, u, oi, ra);

#endif

            l2 = l >> 32;

            l1 = l;

        } else

#endif

        {

            /* Tell the main loop we need to serialize this insn.  */

            cpu_loop_exit_atomic(ENV_GET_CPU(env), ra);

        }

    } else {

        /* We're executing in a serial context -- no need to be atomic.  */

        l1 = cpu_ldl_data_ra(env, a1, ra);

        l2 = cpu_ldl_data_ra(env, a2, ra);

        if (l1 == c1 && l2 == c2) {

            cpu_stl_data_ra(env, a1, u1, ra);

            cpu_stl_data_ra(env, a2, u2, ra);

        }

    }



    if (c1 != l1) {

        env->cc_n = l1;

        env->cc_v = c1;

    } else {

        env->cc_n = l2;

        env->cc_v = c2;

    }

    env->cc_op = CC_OP_CMPL;

    env->dregs[Dc1] = l1;

    env->dregs[Dc2] = l2;

}
