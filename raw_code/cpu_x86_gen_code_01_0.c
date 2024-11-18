int cpu_x86_gen_code(uint8_t *gen_code_buf, int max_code_size, 

                     int *gen_code_size_ptr, uint8_t *pc_start, 

                     int flags)

{

    DisasContext dc1, *dc = &dc1;

    uint8_t *gen_code_end, *pc_ptr;

    long ret;

#ifdef DEBUG_DISAS

    struct disassemble_info disasm_info;

#endif

    dc->code32 = (flags >> GEN_FLAG_CODE32_SHIFT) & 1;

    dc->addseg = (flags >> GEN_FLAG_ADDSEG_SHIFT) & 1;

    dc->f_st = (flags >> GEN_FLAG_ST_SHIFT) & 7;

    dc->cc_op = CC_OP_DYNAMIC;

    gen_code_ptr = gen_code_buf;

    gen_code_end = gen_code_buf + max_code_size - 4096;

    gen_start();



    dc->is_jmp = 0;

    pc_ptr = pc_start;

    do {

        ret = disas_insn(dc, pc_ptr);

        if (ret == -1) 

            error("unknown instruction at PC=0x%x B=%02x %02x", 

                  pc_ptr, pc_ptr[0], pc_ptr[1]);

        pc_ptr = (void *)ret;

    } while (!dc->is_jmp && gen_code_ptr < gen_code_end);

    /* we must store the eflags state if it is not already done */

    if (dc->cc_op != CC_OP_DYNAMIC)

        gen_op_set_cc_op(dc->cc_op);

    if (dc->is_jmp != 1) {

        /* we add an additionnal jmp to update the simulated PC */

        gen_op_jmp_im(ret);

    }

    gen_end();

    *gen_code_size_ptr = gen_code_ptr - gen_code_buf;



#ifdef DEBUG_DISAS

    if (loglevel) {

        uint8_t *pc;

        int count;



        INIT_DISASSEMBLE_INFO(disasm_info, logfile, fprintf);

#if 0        

        disasm_info.flavour = bfd_get_flavour (abfd);

        disasm_info.arch = bfd_get_arch (abfd);

        disasm_info.mach = bfd_get_mach (abfd);

#endif

#ifdef WORDS_BIGENDIAN

        disasm_info.endian = BFD_ENDIAN_BIG;

#else

        disasm_info.endian = BFD_ENDIAN_LITTLE;

#endif        

        fprintf(logfile, "IN:\n");

        disasm_info.buffer = pc_start;

        disasm_info.buffer_vma = (unsigned long)pc_start;

        disasm_info.buffer_length = pc_ptr - pc_start;

        pc = pc_start;

        while (pc < pc_ptr) {

            fprintf(logfile, "0x%08lx:  ", (long)pc);

            count = print_insn_i386((unsigned long)pc, &disasm_info);

            fprintf(logfile, "\n");

            pc += count;

        }

        fprintf(logfile, "\n");

        

        pc = gen_code_buf;

        disasm_info.buffer = pc;

        disasm_info.buffer_vma = (unsigned long)pc;

        disasm_info.buffer_length = *gen_code_size_ptr;

        fprintf(logfile, "OUT: [size=%d]\n", *gen_code_size_ptr);

        while (pc < gen_code_ptr) {

            fprintf(logfile, "0x%08lx:  ", (long)pc);

            count = print_insn_i386((unsigned long)pc, &disasm_info);

            fprintf(logfile, "\n");

            pc += count;

        }

        fprintf(logfile, "\n");

    }

#endif

    return 0;

}
