static inline TranslationBlock *tb_find_fast(void)

{

    TranslationBlock *tb;

    target_ulong cs_base, pc;

    int flags;



    /* we record a subset of the CPU state. It will

       always be the same before a given translated block

       is executed. */

    cpu_get_tb_cpu_state(env, &pc, &cs_base, &flags);

    tb = env->tb_jmp_cache[tb_jmp_cache_hash_func(pc)];

    if (unlikely(!tb || tb->pc != pc || tb->cs_base != cs_base ||

                 tb->flags != flags)) {

        tb = tb_find_slow(pc, cs_base, flags);

    }

    return tb;

}
