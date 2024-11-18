void tlb_fill (target_ulong addr, int is_write, int mmu_idx, void *retaddr)

{

    TranslationBlock *tb;

    CPUState *saved_env;

    unsigned long pc;

    int ret;



    /* XXX: hack to restore env in all cases, even if not called from

       generated code */

    saved_env = env;

    env = cpu_single_env;



    D_LOG("%s pc=%x tpc=%x ra=%x\n", __func__, 

	     env->pc, env->debug1, retaddr);

    ret = cpu_cris_handle_mmu_fault(env, addr, is_write, mmu_idx);

    if (unlikely(ret)) {

        if (retaddr) {

            /* now we have a real cpu fault */

            pc = (unsigned long)retaddr;

            tb = tb_find_pc(pc);

            if (tb) {

                /* the PC is inside the translated code. It means that we have

                   a virtual CPU fault */

                cpu_restore_state(tb, env, pc);



		/* Evaluate flags after retranslation.  */

                helper_top_evaluate_flags();

            }

        }

        cpu_loop_exit(env);

    }

    env = saved_env;

}
