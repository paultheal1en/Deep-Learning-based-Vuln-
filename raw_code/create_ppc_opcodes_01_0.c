static int create_ppc_opcodes (CPUPPCState *env, const ppc_def_t *def)

{

    opcode_t *opc, *start, *end;



    fill_new_table(env->opcodes, 0x40);

    if (&opc_start < &opc_end) {

        start = &opc_start;

        end = &opc_end;

    } else {

        start = &opc_end;

        end = &opc_start;

    }

    for (opc = start + 1; opc != end; opc++) {

        if ((opc->handler.type & def->insns_flags) != 0) {

            if (register_insn(env->opcodes, opc) < 0) {

                printf("*** ERROR initializing PowerPC instruction "

                       "0x%02x 0x%02x 0x%02x\n", opc->opc1, opc->opc2,

                       opc->opc3);

                return -1;

            }

        }

    }

    fix_opcode_tables(env->opcodes);

    fflush(stdout);

    fflush(stderr);



    return 0;

}
