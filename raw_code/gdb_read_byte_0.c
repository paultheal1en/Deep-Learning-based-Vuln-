static void gdb_read_byte(GDBState *s, int ch)

{

    CPUState *env = s->env;

    int i, csum;

    uint8_t reply;



#ifndef CONFIG_USER_ONLY

    if (s->last_packet_len) {

        /* Waiting for a response to the last packet.  If we see the start

           of a new command then abandon the previous response.  */

        if (ch == '-') {

#ifdef DEBUG_GDB

            printf("Got NACK, retransmitting\n");

#endif

            put_buffer(s, (uint8_t *)s->last_packet, s->last_packet_len);

        }

#ifdef DEBUG_GDB

        else if (ch == '+')

            printf("Got ACK\n");

        else

            printf("Got '%c' when expecting ACK/NACK\n", ch);

#endif

        if (ch == '+' || ch == '$')

            s->last_packet_len = 0;

        if (ch != '$')

            return;

    }

    if (vm_running) {

        /* when the CPU is running, we cannot do anything except stop

           it when receiving a char */

        vm_stop(EXCP_INTERRUPT);

    } else

#endif

    {

        switch(s->state) {

        case RS_IDLE:

            if (ch == '$') {

                s->line_buf_index = 0;

                s->state = RS_GETLINE;

            }

            break;

        case RS_GETLINE:

            if (ch == '#') {

            s->state = RS_CHKSUM1;

            } else if (s->line_buf_index >= sizeof(s->line_buf) - 1) {

                s->state = RS_IDLE;

            } else {

            s->line_buf[s->line_buf_index++] = ch;

            }

            break;

        case RS_CHKSUM1:

            s->line_buf[s->line_buf_index] = '\0';

            s->line_csum = fromhex(ch) << 4;

            s->state = RS_CHKSUM2;

            break;

        case RS_CHKSUM2:

            s->line_csum |= fromhex(ch);

            csum = 0;

            for(i = 0; i < s->line_buf_index; i++) {

                csum += s->line_buf[i];

            }

            if (s->line_csum != (csum & 0xff)) {

                reply = '-';

                put_buffer(s, &reply, 1);

                s->state = RS_IDLE;

            } else {

                reply = '+';

                put_buffer(s, &reply, 1);

                s->state = gdb_handle_packet(s, env, s->line_buf);

            }

            break;

        default:

            abort();

        }

    }

}
