static void dec_null(DisasContext *dc)

{







    qemu_log ("unknown insn pc=%x opc=%x\n", dc->pc, dc->opcode);

    dc->abort_at_next_insn = 1;
