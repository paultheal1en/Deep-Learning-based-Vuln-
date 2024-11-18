static void conditional_branch(DBDMA_channel *ch)

{

    dbdma_cmd *current = &ch->current;

    uint16_t br;

    uint16_t sel_mask, sel_value;

    uint32_t status;

    int cond;



    DBDMA_DPRINTF("conditional_branch\n");



    /* check if we must branch */



    br = le16_to_cpu(current->command) & BR_MASK;



    switch(br) {

    case BR_NEVER:  /* don't branch */

        next(ch);

        return;

    case BR_ALWAYS: /* always branch */

        branch(ch);

        return;

    }



    status = be32_to_cpu(ch->regs[DBDMA_STATUS]) & DEVSTAT;



    sel_mask = (be32_to_cpu(ch->regs[DBDMA_BRANCH_SEL]) >> 16) & 0x0f;

    sel_value = be32_to_cpu(ch->regs[DBDMA_BRANCH_SEL]) & 0x0f;



    cond = (status & sel_mask) == (sel_value & sel_mask);



    switch(br) {

    case BR_IFSET:  /* branch if condition bit is 1 */

        if (cond)

            branch(ch);

        else

            next(ch);

        return;

    case BR_IFCLR:  /* branch if condition bit is 0 */

        if (!cond)

            branch(ch);

        else

            next(ch);

        return;

    }

}
