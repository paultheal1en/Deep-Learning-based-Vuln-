static int register_insn (opc_handler_t **ppc_opcodes, opcode_t *insn)

{

    if (insn->opc2 != 0xFF) {

        if (insn->opc3 != 0xFF) {

            if (register_dblind_insn(ppc_opcodes, insn->opc1, insn->opc2,

                                     insn->opc3, &insn->handler) < 0)

                return -1;

        } else {

            if (register_ind_insn(ppc_opcodes, insn->opc1,

                                  insn->opc2, &insn->handler) < 0)

                return -1;

        }

    } else {

        if (register_direct_insn(ppc_opcodes, insn->opc1, &insn->handler) < 0)

            return -1;

    }



    return 0;

}
