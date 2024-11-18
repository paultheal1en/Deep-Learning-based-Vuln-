static void action_command(EEPRO100State *s)

{

    for (;;) {

        bool bit_el;

        bool bit_s;

        bool bit_i;

        bool bit_nc;

        uint16_t ok_status = STATUS_OK;

        s->cb_address = s->cu_base + s->cu_offset;

        read_cb(s);

        bit_el = ((s->tx.command & COMMAND_EL) != 0);

        bit_s = ((s->tx.command & COMMAND_S) != 0);

        bit_i = ((s->tx.command & COMMAND_I) != 0);

        bit_nc = ((s->tx.command & COMMAND_NC) != 0);

#if 0

        bool bit_sf = ((s->tx.command & COMMAND_SF) != 0);

#endif

        s->cu_offset = s->tx.link;

        TRACE(OTHER,

              logout("val=(cu start), status=0x%04x, command=0x%04x, link=0x%08x\n",

                     s->tx.status, s->tx.command, s->tx.link));

        switch (s->tx.command & COMMAND_CMD) {

        case CmdNOp:

            /* Do nothing. */

            break;

        case CmdIASetup:

            cpu_physical_memory_read(s->cb_address + 8, &s->conf.macaddr.a[0], 6);

            TRACE(OTHER, logout("macaddr: %s\n", nic_dump(&s->conf.macaddr.a[0], 6)));

            break;

        case CmdConfigure:

            cpu_physical_memory_read(s->cb_address + 8, &s->configuration[0],

                                     sizeof(s->configuration));

            TRACE(OTHER, logout("configuration: %s\n", nic_dump(&s->configuration[0], 16)));

            break;

        case CmdMulticastList:

            set_multicast_list(s);

            break;

        case CmdTx:

            if (bit_nc) {

                missing("CmdTx: NC = 0");

                ok_status = 0;

                break;

            }

            tx_command(s);

            break;

        case CmdTDR:

            TRACE(OTHER, logout("load microcode\n"));

            /* Starting with offset 8, the command contains

             * 64 dwords microcode which we just ignore here. */

            break;

        case CmdDiagnose:

            TRACE(OTHER, logout("diagnose\n"));

            /* Make sure error flag is not set. */

            s->tx.status = 0;

            break;

        default:

            missing("undefined command");

            ok_status = 0;

            break;

        }

        /* Write new status. */

        stw_phys(s->cb_address, s->tx.status | ok_status | STATUS_C);

        if (bit_i) {

            /* CU completed action. */

            eepro100_cx_interrupt(s);

        }

        if (bit_el) {

            /* CU becomes idle. Terminate command loop. */

            set_cu_state(s, cu_idle);

            eepro100_cna_interrupt(s);

            break;

        } else if (bit_s) {

            /* CU becomes suspended. Terminate command loop. */

            set_cu_state(s, cu_suspended);

            eepro100_cna_interrupt(s);

            break;

        } else {

            /* More entries in list. */

            TRACE(OTHER, logout("CU list with at least one more entry\n"));

        }

    }

    TRACE(OTHER, logout("CU list empty\n"));

    /* List is empty. Now CU is idle or suspended. */

}
