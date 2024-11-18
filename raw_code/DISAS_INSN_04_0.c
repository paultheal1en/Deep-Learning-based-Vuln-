DISAS_INSN(branch)

{

    int32_t offset;

    uint32_t base;

    int op;

    int l1;



    base = s->pc;

    op = (insn >> 8) & 0xf;

    offset = (int8_t)insn;

    if (offset == 0) {

        offset = cpu_ldsw_code(env, s->pc);

        s->pc += 2;

    } else if (offset == -1) {

        offset = read_im32(env, s);

    }

    if (op == 1) {

        /* bsr */

        gen_push(s, tcg_const_i32(s->pc));

    }

    gen_flush_cc_op(s);

    if (op > 1) {

        /* Bcc */

        l1 = gen_new_label();

        gen_jmpcc(s, ((insn >> 8) & 0xf) ^ 1, l1);

        gen_jmp_tb(s, 1, base + offset);

        gen_set_label(l1);

        gen_jmp_tb(s, 0, s->pc);

    } else {

        /* Unconditional branch.  */

        gen_jmp_tb(s, 0, base + offset);

    }

}
