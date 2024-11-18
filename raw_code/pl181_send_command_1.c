static void pl181_send_command(pl181_state *s)

{

    SDRequest request;

    uint8_t response[16];

    int rlen;



    request.cmd = s->cmd & PL181_CMD_INDEX;

    request.arg = s->cmdarg;

    DPRINTF("Command %d %08x\n", request.cmd, request.arg);

    rlen = sd_do_command(s->card, &request, response);

    if (rlen < 0)

        goto error;

    if (s->cmd & PL181_CMD_RESPONSE) {

#define RWORD(n) ((response[n] << 24) | (response[n + 1] << 16) \

                  | (response[n + 2] << 8) | response[n + 3])

        if (rlen == 0 || (rlen == 4 && (s->cmd & PL181_CMD_LONGRESP)))

            goto error;

        if (rlen != 4 && rlen != 16)

            goto error;

        s->response[0] = RWORD(0);

        if (rlen == 4) {

            s->response[1] = s->response[2] = s->response[3] = 0;

        } else {

            s->response[1] = RWORD(4);

            s->response[2] = RWORD(8);

            s->response[3] = RWORD(12) & ~1;

        }

        DPRINTF("Response received\n");

        s->status |= PL181_STATUS_CMDRESPEND;

#undef RWORD

    } else {

        DPRINTF("Command sent\n");

        s->status |= PL181_STATUS_CMDSENT;

    }

    return;



error:

    DPRINTF("Timeout\n");

    s->status |= PL181_STATUS_CMDTIMEOUT;

}
