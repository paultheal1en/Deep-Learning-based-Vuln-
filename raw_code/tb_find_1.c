static inline TranslationBlock *tb_find(CPUState *cpu,

                                        TranslationBlock *last_tb,

                                        int tb_exit)

{

    CPUArchState *env = (CPUArchState *)cpu->env_ptr;

    TranslationBlock *tb;

    target_ulong cs_base, pc;

    uint32_t flags;

    bool acquired_tb_lock = false;



    /* we record a subset of the CPU state. It will

       always be the same before a given translated block

       is executed. */

    cpu_get_tb_cpu_state(env, &pc, &cs_base, &flags);

    tb = atomic_rcu_read(&cpu->tb_jmp_cache[tb_jmp_cache_hash_func(pc)]);

    if (unlikely(!tb || tb->pc != pc || tb->cs_base != cs_base ||

                 tb->flags != flags ||

                 tb->trace_vcpu_dstate != *cpu->trace_dstate)) {

        tb = tb_htable_lookup(cpu, pc, cs_base, flags);

        if (!tb) {



            /* mmap_lock is needed by tb_gen_code, and mmap_lock must be

             * taken outside tb_lock. As system emulation is currently

             * single threaded the locks are NOPs.

             */

            mmap_lock();

            tb_lock();

            acquired_tb_lock = true;



            /* There's a chance that our desired tb has been translated while

             * taking the locks so we check again inside the lock.

             */

            tb = tb_htable_lookup(cpu, pc, cs_base, flags);

            if (!tb) {

                /* if no translated code available, then translate it now */

                tb = tb_gen_code(cpu, pc, cs_base, flags, 0);

            }



            mmap_unlock();

        }



        /* We add the TB in the virtual pc hash table for the fast lookup */

        atomic_set(&cpu->tb_jmp_cache[tb_jmp_cache_hash_func(pc)], tb);

    }

#ifndef CONFIG_USER_ONLY

    /* We don't take care of direct jumps when address mapping changes in

     * system emulation. So it's not safe to make a direct jump to a TB

     * spanning two pages because the mapping for the second page can change.

     */

    if (tb->page_addr[1] != -1) {

        last_tb = NULL;

    }

#endif

    /* See if we can patch the calling TB. */

    if (last_tb && !qemu_loglevel_mask(CPU_LOG_TB_NOCHAIN)) {

        if (!acquired_tb_lock) {

            tb_lock();

            acquired_tb_lock = true;

        }

        if (!tb->invalid) {

            tb_add_jump(last_tb, tb_exit, tb);

        }

    }

    if (acquired_tb_lock) {

        tb_unlock();

    }

    return tb;

}
