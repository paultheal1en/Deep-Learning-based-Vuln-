static int dump_ppc_insns (CPUPPCState *env)

{

    opc_handler_t **table, *handler;

    uint8_t opc1, opc2, opc3;



    printf("Instructions set:\n");

    /* opc1 is 6 bits long */

    for (opc1 = 0x00; opc1 < 0x40; opc1++) {

        table = env->opcodes;

        handler = table[opc1];

        if (is_indirect_opcode(handler)) {

            /* opc2 is 5 bits long */

            for (opc2 = 0; opc2 < 0x20; opc2++) {

                table = env->opcodes;

                handler = env->opcodes[opc1];

                table = ind_table(handler);

                handler = table[opc2];

                if (is_indirect_opcode(handler)) {

                    table = ind_table(handler);

                    /* opc3 is 5 bits long */

                    for (opc3 = 0; opc3 < 0x20; opc3++) {

                        handler = table[opc3];

                        if (handler->handler != &gen_invalid) {

                            printf("INSN: %02x %02x %02x (%02d %04d) : %s\n",

                                   opc1, opc2, opc3, opc1, (opc3 << 5) | opc2,

                                   handler->oname);

                        }

                    }

                } else {

                    if (handler->handler != &gen_invalid) {

                        printf("INSN: %02x %02x -- (%02d %04d) : %s\n",

                               opc1, opc2, opc1, opc2, handler->oname);

                    }

                }

            }

        } else {

            if (handler->handler != &gen_invalid) {

                printf("INSN: %02x -- -- (%02d ----) : %s\n",

                       opc1, opc1, handler->oname);

            }

        }

    }

}
