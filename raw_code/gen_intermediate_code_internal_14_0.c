static inline int gen_intermediate_code_internal(CPUState *env,

                                                 TranslationBlock *tb, 

                                                 int search_pc)

{

    DisasContext dc1, *dc = &dc1;

    uint8_t *pc_ptr;

    uint16_t *gen_opc_end;

    int flags, j, lj;

    uint8_t *pc_start;

    uint8_t *cs_base;

    

    /* generate intermediate code */

    pc_start = (uint8_t *)tb->pc;

    cs_base = (uint8_t *)tb->cs_base;

    flags = tb->flags;

       

    dc->pe = env->cr[0] & CR0_PE_MASK;

    dc->code32 = (flags >> HF_CS32_SHIFT) & 1;

    dc->ss32 = (flags >> HF_SS32_SHIFT) & 1;

    dc->addseg = (flags >> HF_ADDSEG_SHIFT) & 1;

    dc->f_st = 0;

    dc->vm86 = (flags >> VM_SHIFT) & 1;

    dc->cpl = (flags >> HF_CPL_SHIFT) & 3;

    dc->iopl = (flags >> IOPL_SHIFT) & 3;

    dc->tf = (flags >> TF_SHIFT) & 1;

    dc->singlestep_enabled = env->singlestep_enabled;

    dc->cc_op = CC_OP_DYNAMIC;

    dc->cs_base = cs_base;

    dc->tb = tb;

    dc->popl_esp_hack = 0;

    /* select memory access functions */

    dc->mem_index = 0;

    if (flags & HF_SOFTMMU_MASK) {

        if (dc->cpl == 3)

            dc->mem_index = 6;

        else

            dc->mem_index = 3;

    }

    dc->jmp_opt = !(dc->tf || env->singlestep_enabled

#ifndef CONFIG_SOFT_MMU

                    || (flags & HF_SOFTMMU_MASK)

#endif

                    );

    gen_opc_ptr = gen_opc_buf;

    gen_opc_end = gen_opc_buf + OPC_MAX_SIZE;

    gen_opparam_ptr = gen_opparam_buf;



    dc->is_jmp = DISAS_NEXT;

    pc_ptr = pc_start;

    lj = -1;



    /* if irq were inhibited for the next instruction, we can disable

       them here as it is simpler (otherwise jumps would have to

       handled as special case) */

    if (flags & HF_INHIBIT_IRQ_MASK) {

        gen_op_reset_inhibit_irq();

    }

    for(;;) {

        if (env->nb_breakpoints > 0) {

            for(j = 0; j < env->nb_breakpoints; j++) {

                if (env->breakpoints[j] == (unsigned long)pc_ptr) {

                    gen_debug(dc, pc_ptr - dc->cs_base);

                    break;

                }

            }

        }

        if (search_pc) {

            j = gen_opc_ptr - gen_opc_buf;

            if (lj < j) {

                lj++;

                while (lj < j)

                    gen_opc_instr_start[lj++] = 0;

            }

            gen_opc_pc[lj] = (uint32_t)pc_ptr;

            gen_opc_cc_op[lj] = dc->cc_op;

            gen_opc_instr_start[lj] = 1;

        }

        pc_ptr = disas_insn(dc, pc_ptr);

        /* stop translation if indicated */

        if (dc->is_jmp)

            break;

        /* if single step mode, we generate only one instruction and

           generate an exception */

        if (dc->tf || dc->singlestep_enabled) {

            gen_op_jmp_im(pc_ptr - dc->cs_base);

            gen_eob(dc);

            break;

        }

        /* if too long translation, stop generation too */

        if (gen_opc_ptr >= gen_opc_end ||

            (pc_ptr - pc_start) >= (TARGET_PAGE_SIZE - 32)) {

            gen_op_jmp_im(pc_ptr - dc->cs_base);

            gen_eob(dc);

            break;

        }

    }

    *gen_opc_ptr = INDEX_op_end;

    /* we don't forget to fill the last values */

    if (search_pc) {

        j = gen_opc_ptr - gen_opc_buf;

        lj++;

        while (lj <= j)

            gen_opc_instr_start[lj++] = 0;

    }

        

#ifdef DEBUG_DISAS

    if (loglevel) {

        fprintf(logfile, "----------------\n");

        fprintf(logfile, "IN: %s\n", lookup_symbol(pc_start));

	disas(logfile, pc_start, pc_ptr - pc_start, 0, !dc->code32);

        fprintf(logfile, "\n");



        fprintf(logfile, "OP:\n");

        dump_ops(gen_opc_buf, gen_opparam_buf);

        fprintf(logfile, "\n");

    }

#endif



    /* optimize flag computations */

    optimize_flags(gen_opc_buf, gen_opc_ptr - gen_opc_buf);



#ifdef DEBUG_DISAS

    if (loglevel) {

        fprintf(logfile, "AFTER FLAGS OPT:\n");

        dump_ops(gen_opc_buf, gen_opparam_buf);

        fprintf(logfile, "\n");

    }

#endif

    if (!search_pc)

        tb->size = pc_ptr - pc_start;

    return 0;

}
