int cpu_restore_state(TranslationBlock *tb,

                      CPUState *env, unsigned long searched_pc,

                      void *puc)

{

    TCGContext *s = &tcg_ctx;

    int j;

    unsigned long tc_ptr;

#ifdef CONFIG_PROFILER

    int64_t ti;

#endif



#ifdef CONFIG_PROFILER

    ti = profile_getclock();

#endif

    tcg_func_start(s);



    if (gen_intermediate_code_pc(env, tb) < 0)

        return -1;



    /* find opc index corresponding to search_pc */

    tc_ptr = (unsigned long)tb->tc_ptr;

    if (searched_pc < tc_ptr)

        return -1;



    s->tb_next_offset = tb->tb_next_offset;

#ifdef USE_DIRECT_JUMP

    s->tb_jmp_offset = tb->tb_jmp_offset;

    s->tb_next = NULL;

#else

    s->tb_jmp_offset = NULL;

    s->tb_next = tb->tb_next;

#endif

    j = dyngen_code_search_pc(s, (uint8_t *)tc_ptr, 

                              (void *)searched_pc); 

    if (j < 0)

        return -1;

    /* now find start of instruction before */

    while (gen_opc_instr_start[j] == 0)

        j--;

#if defined(TARGET_I386)

    {

        int cc_op;

#ifdef DEBUG_DISAS

        if (loglevel & CPU_LOG_TB_OP) {

            int i;

            fprintf(logfile, "RESTORE:\n");

            for(i=0;i<=j; i++) {

                if (gen_opc_instr_start[i]) {

                    fprintf(logfile, "0x%04x: " TARGET_FMT_lx "\n", i, gen_opc_pc[i]);

                }

            }

            fprintf(logfile, "spc=0x%08lx j=0x%x eip=" TARGET_FMT_lx " cs_base=%x\n",

                    searched_pc, j, gen_opc_pc[j] - tb->cs_base,

                    (uint32_t)tb->cs_base);

        }

#endif

        env->eip = gen_opc_pc[j] - tb->cs_base;

        cc_op = gen_opc_cc_op[j];

        if (cc_op != CC_OP_DYNAMIC)

            env->cc_op = cc_op;

    }

#elif defined(TARGET_ARM)

    env->regs[15] = gen_opc_pc[j];

#elif defined(TARGET_SPARC)

    {

        target_ulong npc;

        env->pc = gen_opc_pc[j];

        npc = gen_opc_npc[j];

        if (npc == 1) {

            /* dynamic NPC: already stored */

        } else if (npc == 2) {

            target_ulong t2 = (target_ulong)(unsigned long)puc;

            /* jump PC: use T2 and the jump targets of the translation */

            if (t2)

                env->npc = gen_opc_jump_pc[0];

            else

                env->npc = gen_opc_jump_pc[1];

        } else {

            env->npc = npc;

        }

    }

#elif defined(TARGET_PPC)

    {

        int type, c;

        /* for PPC, we need to look at the micro operation to get the

           access type */

        env->nip = gen_opc_pc[j];

        c = gen_opc_buf[j];

        switch(c) {

#if defined(CONFIG_USER_ONLY)

#define CASE3(op)\

        case INDEX_op_ ## op ## _raw

#else

#define CASE3(op)\

        case INDEX_op_ ## op ## _user:\

        case INDEX_op_ ## op ## _kernel:\

        case INDEX_op_ ## op ## _hypv

#endif



        CASE3(stfd):

        CASE3(stfs):

        CASE3(lfd):

        CASE3(lfs):

            type = ACCESS_FLOAT;

            break;

        CASE3(lwarx):

            type = ACCESS_RES;

            break;

        CASE3(stwcx):

            type = ACCESS_RES;

            break;

        CASE3(eciwx):

        CASE3(ecowx):

            type = ACCESS_EXT;

            break;

        default:

            type = ACCESS_INT;

            break;

        }

        env->access_type = type;

    }

#elif defined(TARGET_M68K)

    env->pc = gen_opc_pc[j];

#elif defined(TARGET_MIPS)

    env->PC[env->current_tc] = gen_opc_pc[j];

    env->hflags &= ~MIPS_HFLAG_BMASK;

    env->hflags |= gen_opc_hflags[j];

#elif defined(TARGET_ALPHA)

    env->pc = gen_opc_pc[j];

#elif defined(TARGET_SH4)

    env->pc = gen_opc_pc[j];

    env->flags = gen_opc_hflags[j];

#endif



#ifdef CONFIG_PROFILER

    dyngen_restore_time += profile_getclock() - ti;

    dyngen_restore_count++;

#endif

    return 0;

}
