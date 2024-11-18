void tlb_fill(target_ulong addr, int is_write, int is_user, void *retaddr)

{

    TranslationBlock *tb;

    int ret;

    unsigned long pc;

    CPUX86State *saved_env;



    /* XXX: hack to restore env in all cases, even if not called from

       generated code */

    saved_env = env;

    env = cpu_single_env;



    ret = cpu_x86_handle_mmu_fault(env, addr, is_write, is_user, 1);

    if (ret) {

        if (retaddr) {

            /* now we have a real cpu fault */

            pc = (unsigned long)retaddr;

            tb = tb_find_pc(pc);

            if (tb) {

                /* the PC is inside the translated code. It means that we have

                   a virtual CPU fault */

                cpu_restore_state(tb, env, pc, NULL);

            }

        }

        if (retaddr)

            raise_exception_err(EXCP0E_PAGE, env->error_code);

        else

            raise_exception_err_norestore(EXCP0E_PAGE, env->error_code);

    }

    env = saved_env;

}
